"""Bake-off M4-v4 parallel: 全7 config を CUDA ストリームで真並列実行

v4 からの変更点:
  - 7 config を ThreadPoolExecutor + CUDA Stream で同時訓練
  - データは 1 回だけロード（全 config で共有）
  - DataLoader num_workers=4 per config（合計 28 core 活用）
  - 28 CPU コア / 7 config = 4 workers ずつ
  - GPU: 7 config × ~600 MiB = ~4.2 GB（RTX5090 32GB に余裕）

期待速度: ~7-8h → ~1.5-2h（7倍近い高速化）
"""
from __future__ import annotations
import json, sys, os, time, threading, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
# ThreadPoolExecutor内で複数のDataLoaderを動かすとファイルディスクリプタが枯渇するため
# file_system共有戦略を使用（共有メモリFDの代わりにファイルを使う）
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, log_loss,
)
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from src.config import load_and_prepare
from src.utils import seed_everything
from feature_loader import build_meta_lookup, build_speaking_mask, MOUTH_AU_COLS

# ── ハイパーパラメータ ────────────────────────────────────────────
LA_TAU        = float(os.environ.get("LA_TAU",   "1.0"))
MOD_DROP_P    = float(os.environ.get("MOD_DROP", "0.3"))
MODEL_NAME    = "M4_attmil_late_v4"
N_OUTER_TRIALS = int(os.environ.get("N_TRIALS", "10"))
N_INNER_SPLITS = 3
N_CLASSES      = 3
N_BINARY       = 2
PATIENCE       = 8
HZ             = 20
N_WORKERS      = 2   # DataLoader workers per config thread (4→2: FD枯渇対策)

# 7 config 定義 (config名: (zero_face, zero_voice, zero_text))
ALL_CONFIGS = {
    "Full":       (False, False, False),
    "Text-only":  (True,  True,  False),
    "Audio-only": (True,  False, True),
    "Face-only":  (False, True,  True),
    "Audio+Text": (True,  False, False),
    "Face+Text":  (False, True,  False),
    "Audio+Face": (False, False, True),
}

# ── 早期離脱ユーザ除外 (★6) ──────────────────────────────────────
# 記録期間 (first-to-last recording span) が MIN_DAYS 未満のユーザを除外する。
# v4 では MIN_DAYS=15（=「≥15 日」基準）。
# 旧版は ID をハードコードしていたが、計算ベースに変更して再現性・匿名性を向上。
MIN_RECORDING_DAYS = int(os.environ.get("MIN_RECORDING_DAYS", 15))

def _compute_dropout_users(df, min_days=MIN_RECORDING_DAYS, ts_col="video_id"):
    """Return the set of user_ids whose first-to-last recording span is < min_days.

    The timestamp is parsed from the trailing YYYYMMDDHHMMSS in ``video_id``
    (e.g., ``S00101_20240217171842``). Returns an empty set when min_days <= 0.
    """
    if min_days <= 0:
        return set()
    ts = pd.to_datetime(df[ts_col].str.extract(r"_(\d{14})$")[0], format="%Y%m%d%H%M%S")
    span = (ts.groupby(df["user_id"]).max() - ts.groupby(df["user_id"]).min()).dt.days
    return set(span[span < min_days].index)

SMOKE         = int(os.environ.get("SMOKE", 0))
if SMOKE:
    N_OUTER_TRIALS = 3

cfg, _, OUT_DIR = load_and_prepare("/workspace/config.yaml")
SEED = int(cfg.get("runtime", {}).get("seed", 42))
seed_everything(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

seq_dir  = OUT_DIR / "sequences"
seg_dir  = seq_dir / "seg_text"
SAVE_DIR = Path("/workspace/revision/analyses/bakeoff")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── データ読み込み ────────────────────────────────────────────────
idx_all = pd.read_parquet(seq_dir / "seq_index.parquet").reset_index(drop=True)

# complete-dispersion 除外
# Note: label_dok / label_nis / label_srk are opaque column identifiers
# preserved for backwards compatibility with the upstream parquet schema;
# they correspond to anonymised psychiatrist raters R1 / R2 / R3 in the
# manuscript and supporting information.
def _complete_dispersion(row):
    vals = [row.get("label_dok"), row.get("label_nis"), row.get("label_srk")]
    vals = [int(v) for v in vals if pd.notna(v) and int(v) in (1, 2, 3)]
    return len(vals) == 3 and len(set(vals)) == 3
disp_mask = idx_all.apply(_complete_dispersion, axis=1)
if disp_mask.sum() > 0:
    print(f"[{MODEL_NAME}] Excluding {disp_mask.sum()} complete-dispersion clips", flush=True)
    idx_all = idx_all[~disp_mask].reset_index(drop=True)

# ★6 早期離脱除外（記録期間が MIN_RECORDING_DAYS 未満のユーザを動的に検出）
dropout_users = _compute_dropout_users(idx_all, MIN_RECORDING_DAYS)
dropout_mask = idx_all["user_id"].isin(dropout_users)
if dropout_mask.sum() > 0:
    print(f"[{MODEL_NAME}] ★6 Excluding {dropout_mask.sum()} clips from "
          f"{len(dropout_users)} users with recording span < {MIN_RECORDING_DAYS} days "
          f"({len(idx_all)} → {len(idx_all)-dropout_mask.sum()})", flush=True)
    idx_all = idx_all[~dropout_mask].reset_index(drop=True)

# splits
splits_path = seq_dir / "splits_repeats_v4_ncd.json"
if not splits_path.exists():
    print(f"[{MODEL_NAME}] Building splits from {len(idx_all)} clips ...", flush=True)
    from sklearn.model_selection import GroupKFold as _GKF
    _R, _K, _SEED0 = 5, 5, 42
    _groups = idx_all["user_id"].astype(str).to_numpy()
    _unique_users = np.unique(_groups)
    _repeats = []
    for _r in range(_R):
        _rng = np.random.RandomState(_SEED0 + _r)
        _perm = _rng.permutation(_unique_users)
        _u2f = {u: int(i % _K) for i, u in enumerate(_perm)}
        _sfold = np.array([_u2f[g] for g in _groups], dtype=np.int64)
        _folds = []
        for _k in range(_K):
            _te = np.where(_sfold == _k)[0]
            _tr = np.where(_sfold != _k)[0]
            _folds.append({"train_idx": _tr.tolist(), "val_idx": _te.tolist()})
        _repeats.append({"folds": _folds, "seed": int(_SEED0 + _r)})
    _obj = {"repeats": _repeats,
            "meta": {"R": _R, "K_outer": _K, "seed0": _SEED0,
                     "scheme": "group_permuted_kfold",
                     "note": "complete_dispersion_excluded+early_dropout_excluded_le14days"}}
    splits_path.write_text(json.dumps(_obj, indent=2))
    print(f"[{MODEL_NAME}] Splits saved → {splits_path}", flush=True)
rep_obj = json.loads(splits_path.read_text())

print(f"[{MODEL_NAME}] {len(idx_all)} clips | device={device} | "
      f"GPU={torch.cuda.get_device_name(0) if device=='cuda' else 'N/A'}", flush=True)

meta_lookup = build_meta_lookup(idx_all, cfg)

# ── MIL データロード ─────────────────────────────────────────────
INST_DIM = 812
FACE_SL  = slice(0, 22)
VOICE_SL = slice(22, 44)
TEXT_SL  = slice(44, 812)

def _seg_mean_fv(seq_fv, t_start, t_end):
    f0 = max(0, int(round(float(t_start) * HZ)))
    f1 = max(0, int(round(float(t_end)   * HZ)))
    f1 = min(f1, seq_fv.shape[0])
    if f1 <= f0:
        return seq_fv[f0:f0+1].mean(0) if f0 < seq_fv.shape[0] else np.zeros(seq_fv.shape[1])
    return seq_fv[f0:f1].mean(0)

def load_clip_mil(row):
    npz   = np.load(row["seq_path"], allow_pickle=True)
    seq   = npz["seq"].astype(np.float32)
    mask  = npz["mask"].astype(np.float32)
    label = int(npz["label"])
    vid   = row["video_id"]
    face_full  = seq[:, 0:22]
    voice_full = seq[:, 22:44]
    seg_path = seg_dir / f"{vid}_seg.npz"
    if not seg_path.exists():
        valid_t = mask > 0
        f = face_full[valid_t].mean(0) if valid_t.any() else np.zeros(22)
        v = voice_full[valid_t].mean(0) if valid_t.any() else np.zeros(22)
        t = seq[valid_t, 44:812].mean(0) if valid_t.any() else np.zeros(768)
        return np.concatenate([f, v, t])[None].astype(np.float32), label
    sg      = np.load(str(seg_path), allow_pickle=True)
    X_text  = sg["X_text"].astype(np.float32)
    t_start = sg["t_start"].astype(np.float64)
    t_end   = sg.get("t_end", t_start + 3.0).astype(np.float64)
    seg_msk = sg["mask"].astype(np.float32) if "mask" in sg else np.ones(len(X_text))
    instances = []
    for s in range(len(X_text)):
        if seg_msk[s] <= 0: continue
        f_seg = _seg_mean_fv(face_full,  t_start[s], t_end[s])
        v_seg = _seg_mean_fv(voice_full, t_start[s], t_end[s])
        instances.append(np.concatenate([f_seg, v_seg, X_text[s]]))
    if len(instances) == 0:
        valid_t = mask > 0
        f = face_full[valid_t].mean(0) if valid_t.any() else np.zeros(22)
        v = voice_full[valid_t].mean(0) if valid_t.any() else np.zeros(22)
        instances.append(np.concatenate([f, v, np.zeros(768)]))
    return np.stack(instances).astype(np.float32), label

print(f"[{MODEL_NAME}] Pre-loading {len(idx_all)} clips into RAM...", flush=True)
t_load = time.time()
cache_all = []
for i in range(len(idx_all)):
    row = idx_all.iloc[i]
    try:
        instances, label = load_clip_mil(row)
    except Exception as e:
        instances = np.zeros((1, INST_DIM), dtype=np.float32)
        label = 0
    cache_all.append((instances, label))
    if i % 500 == 0:
        print(f"  {i}/{len(idx_all)} cached...", flush=True)
idx_all = idx_all.copy()
idx_all["_cache_idx"] = np.arange(len(idx_all))
print(f"[{MODEL_NAME}] Data loaded in {time.time()-t_load:.1f}s", flush=True)

# ── Dataset / Model ──────────────────────────────────────────────
class MILDataset(Dataset):
    def __init__(self, df, zero_face, zero_voice, zero_text):
        self.df = df.reset_index(drop=True)
        self.zf, self.zv, self.zt = zero_face, zero_voice, zero_text

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        instances, label = cache_all[int(row["_cache_idx"])]
        inst_t = torch.tensor(instances, dtype=torch.float32)
        if self.zf: inst_t[:, FACE_SL]  = 0.0
        if self.zv: inst_t[:, VOICE_SL] = 0.0
        if self.zt: inst_t[:, TEXT_SL]  = 0.0
        return inst_t, label

def collate_mil(batch):
    insts, labels = zip(*batch)
    B    = len(insts)
    maxS = max(x.shape[0] for x in insts)
    pI   = torch.zeros(B, maxS, INST_DIM)
    mI   = torch.zeros(B, maxS)
    for i in range(B):
        S = insts[i].shape[0]
        pI[i, :S] = insts[i]; mI[i, :S] = 1
    return pI, mI, torch.tensor(labels, dtype=torch.long)

class AttMILStream(nn.Module):
    def __init__(self, in_dim, hidden, dropout):
        super().__init__()
        self.enc    = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.attn_V = nn.Linear(hidden, hidden, bias=False)
        self.attn_U = nn.Linear(hidden, hidden, bias=False)
        self.attn_w = nn.Linear(hidden, 1, bias=False)
        self.head   = nn.Linear(hidden, N_BINARY)

    def forward(self, x, mask):
        h = self.enc(x)
        a = self.attn_w(torch.tanh(self.attn_V(h) * torch.sigmoid(self.attn_U(h)))).squeeze(-1)
        a = a.masked_fill(mask == 0, -1e9)
        a = torch.softmax(a, dim=-1)
        z = (a.unsqueeze(-1) * h).sum(dim=1)
        return self.head(z), a

class AttMILLate(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.face_mil  = AttMILStream(22,  hidden, dropout)
        self.voice_mil = AttMILStream(22,  hidden, dropout)
        self.text_mil  = AttMILStream(768, hidden, dropout)
        self.log_w     = nn.Parameter(torch.zeros(3))

    def forward(self, x, mask):
        lf, af = self.face_mil(x[:, :, FACE_SL],  mask)
        lv, av = self.voice_mil(x[:, :, VOICE_SL], mask)
        lt, at = self.text_mil(x[:, :, TEXT_SL],  mask)
        w = torch.softmax(self.log_w, dim=0)
        return w[0]*lf + w[1]*lv + w[2]*lt, lf, lv, lt, af, av, at

def corn_decode_probs(z):
    s0 = torch.sigmoid(z[:, 0]); s1 = torch.sigmoid(z[:, 1])
    return torch.stack([1.0-s0, s0*(1.0-s1), s0*s1], dim=-1)

def corn_decode_probs_np(z):
    s0 = 1/(1+np.exp(-z[:,0])); s1 = 1/(1+np.exp(-z[:,1]))
    return np.stack([1-s0, s0*(1-s1), s0*s1], axis=-1)

def eval_metrics(pr, y):
    p = pr.argmax(1)
    cm = confusion_matrix(y, p, labels=[0,1,2])
    try:   auroc = roc_auc_score(y, pr, multi_class="ovr", average="macro")
    except: auroc = np.nan
    try:   auprc = float(np.nanmean([average_precision_score((y==c).astype(int), pr[:,c]) for c in range(3)]))
    except: auprc = np.nan
    try:   ll = log_loss(y, pr, labels=[0,1,2])
    except: ll = np.nan
    f1p = f1_score(y, p, average=None, labels=[0,1,2], zero_division=0)
    pp  = precision_score(y, p, average=None, labels=[0,1,2], zero_division=0)
    rp  = recall_score(y, p, average=None, labels=[0,1,2], zero_division=0)
    return {
        "f1_macro": float(f1_score(y,p,average="macro")),
        "f1_micro": float(f1_score(y,p,average="micro")),
        "f1_class0": float(f1p[0]), "f1_class1": float(f1p[1]), "f1_class2": float(f1p[2]),
        "prec_macro": float(precision_score(y,p,average="macro",zero_division=0)),
        "rec_macro":  float(recall_score(y,p,average="macro",zero_division=0)),
        "prec_class0": float(pp[0]), "prec_class1": float(pp[1]), "prec_class2": float(pp[2]),
        "rec_class0":  float(rp[0]), "rec_class1":  float(rp[1]), "rec_class2":  float(rp[2]),
        "auroc_macro": auroc, "auprc_macro": auprc, "logloss": ll,
        "mae_ordinal": float(np.mean(np.abs(y-p))),
        "overpred_ratio_class2": float((p==2).sum()/max((y==2).sum(),1)),
        "cm_00":int(cm[0,0]),"cm_01":int(cm[0,1]),"cm_02":int(cm[0,2]),
        "cm_10":int(cm[1,0]),"cm_11":int(cm[1,1]),"cm_12":int(cm[1,2]),
        "cm_20":int(cm[2,0]),"cm_21":int(cm[2,1]),"cm_22":int(cm[2,2]),
    }

# ── スレッドごとの CUDA ストリーム ────────────────────────────────
_thread_streams: dict[str, torch.cuda.Stream] = {}
_stream_lock = threading.Lock()

def get_stream(config_name: str) -> torch.cuda.Stream:
    with _stream_lock:
        if config_name not in _thread_streams:
            _thread_streams[config_name] = torch.cuda.Stream()
    return _thread_streams[config_name]

# ── チェックポイント（スレッドセーフ） ───────────────────────────
_ckpt_locks: dict[str, threading.Lock] = {cfg: threading.Lock() for cfg in ALL_CONFIGS}

def get_ckpt_path(config_name: str) -> Path:
    fn = config_name.replace("+", "_").replace(" ", "_")
    return SAVE_DIR / f"{MODEL_NAME}_{fn}_ckpt.csv"

def load_checkpoint(config_name: str):
    path = get_ckpt_path(config_name)
    if not path.exists():
        return set(), []
    df_ck = pd.read_csv(path)
    done = set(zip(df_ck["repeat"].tolist(), df_ck["fold"].tolist()))
    return done, df_ck.to_dict("records")

def save_checkpoint(config_name: str, metrics: dict):
    path = get_ckpt_path(config_name)
    with _ckpt_locks[config_name]:
        header = not path.exists()
        pd.DataFrame([metrics]).to_csv(path, mode="a", header=header, index=False)

# ── 訓練関数（1 config × 1 fold） ────────────────────────────────
def train_eval_fold(df_tr, df_te, log_biases_t, class_weights_t,
                    hidden, dropout, lr, bs, epochs,
                    zero_face, zero_voice, zero_text,
                    stream, trial=None, step_counter=None, local_seed=42):
    seed_everything(local_seed)
    model = AttMILLate(hidden=hidden, dropout=dropout).to(device)
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    lb_dev = log_biases_t.to(device)
    cw_dev = class_weights_t.to(device)

    def la_bce_corn(logits, labels):
        total = torch.tensor(0.0, device=logits.device)
        for k in range(N_BINARY):
            cond = labels >= k
            if cond.sum() == 0: continue
            z_k = logits[cond, k] + lb_dev[k]
            y_k = (labels[cond] > k).float()
            total = total + F.binary_cross_entropy_with_logits(z_k, y_k)
        return total / N_BINARY

    def coral_bce_aux(logits, labels):
        w = cw_dev[labels]
        total = torch.tensor(0.0, device=logits.device)
        for k in range(N_BINARY):
            y_k = (labels > k).float()
            total = total + F.binary_cross_entropy_with_logits(logits[:,k], y_k, weight=w)
        return total / N_BINARY

    dl_tr = DataLoader(MILDataset(df_tr, zero_face, zero_voice, zero_text),
                       batch_size=int(bs), shuffle=True,
                       collate_fn=collate_mil, num_workers=N_WORKERS,
                       persistent_workers=True)
    dl_te = DataLoader(MILDataset(df_te, zero_face, zero_voice, zero_text),
                       batch_size=int(bs), shuffle=False,
                       collate_fn=collate_mil, num_workers=N_WORKERS,
                       persistent_workers=True)

    best_f1, best_st, no_imp = -1.0, None, 0

    with torch.cuda.stream(stream):
        for ep in range(epochs):
            model.train()
            for inst, mask, lab in dl_tr:
                inst = inst.to(device); mask = mask.to(device); lab = lab.to(device)
                if MOD_DROP_P > 0:
                    if not zero_face:
                        drop_f = torch.rand(inst.shape[0], device=device) < MOD_DROP_P
                        inst[drop_f, :, FACE_SL] = 0.0
                    if not zero_voice:
                        drop_v = torch.rand(inst.shape[0], device=device) < MOD_DROP_P
                        inst[drop_v, :, VOICE_SL] = 0.0
                opt.zero_grad()
                fused, lf, lv, lt, *_ = model(inst, mask)
                loss = la_bce_corn(fused, lab) + 0.2 * coral_bce_aux(lt, lab)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            ys, probs = [], []
            with torch.no_grad():
                for inst, mask, lab in dl_te:
                    fused, *_ = model(inst.to(device), mask.to(device))
                    probs.append(corn_decode_probs(fused).cpu().numpy())
                    ys.append(lab.numpy())
            pr_va = np.vstack(probs); y_va = np.concatenate(ys)
            f1 = f1_score(y_va, pr_va.argmax(1), average="macro", zero_division=0)

            if trial is not None and step_counter is not None:
                trial.report(f1, step_counter[0]); step_counter[0] += 1
                if trial.should_prune(): raise optuna.exceptions.TrialPruned()

            if f1 > best_f1:
                best_f1 = f1
                best_st = {k: v.cpu() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= PATIENCE: break

    stream.synchronize()
    model.load_state_dict(best_st); model.eval()
    ys, probs = [], []
    with torch.no_grad(), torch.cuda.stream(stream):
        for inst, mask, lab in dl_te:
            fused, *_ = model(inst.to(device), mask.to(device))
            probs.append(corn_decode_probs(fused).cpu().numpy())
            ys.append(lab.numpy())
    stream.synchronize()
    return np.vstack(probs), np.concatenate(ys)

# ── 1 config × 1 fold のワーカー関数 ─────────────────────────────
def run_one(config_name: str, zero_face: bool, zero_voice: bool, zero_text: bool,
            r_i: int, f_i: int, tr_i: list, te_i: list,
            done_folds: set, total_folds: int) -> dict | None:

    fold_num = r_i * 5 + f_i + 1
    if (r_i + 1, f_i + 1) in done_folds:
        print(f"  [{config_name}] Fold {fold_num}/{total_folds} SKIP", flush=True)
        return None

    stream = get_stream(config_name)
    local_seed = SEED + hash(config_name) % 10000 + r_i * 100 + f_i

    df_tr = idx_all.iloc[tr_i].reset_index(drop=True)
    df_te = idx_all.iloc[te_i].reset_index(drop=True)
    gtr   = df_tr["user_id"].astype(str).to_numpy()
    ytr   = df_tr["label"].to_numpy()

    # LA biases
    log_biases_np = np.zeros(N_BINARY, dtype=np.float32)
    for k in range(N_BINARY):
        cond_k = ytr >= k
        n_cond = cond_k.sum()
        if n_cond == 0: continue
        n_pos = (ytr[cond_k] > k).sum()
        pi_pos = max(n_pos / n_cond, 1e-6); pi_neg = max(1 - pi_pos, 1e-6)
        log_biases_np[k] = LA_TAU * (np.log(pi_pos) - np.log(pi_neg))
    log_biases_t = torch.tensor(log_biases_np)

    # class weights
    train_counts = np.bincount(ytr, minlength=N_CLASSES).astype(float)
    train_counts = np.where(train_counts == 0, 1.0, train_counts)
    inv = 1.0 / train_counts
    cw  = inv / inv.sum() * N_CLASSES
    class_weights_t = torch.tensor(cw, dtype=torch.float32)

    from sklearn.model_selection import GroupKFold
    u_uniq_tr = np.unique(gtr)
    n_inner = max(2, min(N_INNER_SPLITS, len(u_uniq_tr)))

    def objective(trial):
        seed_everything(local_seed + trial.number)
        hidden  = trial.suggest_categorical("hidden",  [64, 128, 256])
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        lr      = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        bs      = trial.suggest_categorical("bs", [8, 16, 24])
        epochs  = trial.suggest_int("epochs", 10, 30)
        rng_in  = np.random.RandomState(local_seed + trial.number)
        perm_u  = rng_in.permutation(u_uniq_tr)
        u2r     = {u: i for i, u in enumerate(perm_u)}
        gtr_p   = np.array([u2r[g] for g in gtr], dtype=np.int64)
        gkf     = GroupKFold(n_splits=n_inner)
        scores  = []; step_counter = [0]
        for tri, vai in gkf.split(np.zeros(len(df_tr)), ytr, gtr_p):
            pr_va, y_va = train_eval_fold(
                df_tr.iloc[tri], df_tr.iloc[vai],
                log_biases_t, class_weights_t,
                hidden, dropout, lr, bs, epochs,
                zero_face, zero_voice, zero_text,
                stream, trial=trial, step_counter=step_counter,
                local_seed=local_seed + trial.number)
            scores.append(f1_score(y_va, pr_va.argmax(1), average="macro", zero_division=0))
        return float(np.mean(scores))

    t0 = time.time()
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=local_seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=N_OUTER_TRIALS, show_progress_bar=False)

    bp = study.best_params
    pr, yte = train_eval_fold(
        df_tr, df_te, log_biases_t, class_weights_t,
        hidden=int(bp["hidden"]), dropout=float(bp["dropout"]),
        lr=float(bp["lr"]), bs=int(bp["bs"]), epochs=int(bp["epochs"]),
        zero_face=zero_face, zero_voice=zero_voice, zero_text=zero_text,
        stream=stream, local_seed=local_seed)

    # NPZ 保存
    fn = config_name.replace("+", "_").replace(" ", "_")
    vid_arr  = idx_all.iloc[te_i]["video_id"].to_numpy()
    npz_path = SAVE_DIR / f"{MODEL_NAME}_{fn}_r{r_i+1}_f{f_i+1}_preds.npz"
    np.savez_compressed(str(npz_path), pr=pr.astype(np.float32),
                        y=yte.astype(np.int64), video_ids=vid_arr)

    metrics = eval_metrics(pr, yte)
    metrics.update({
        "model": MODEL_NAME, "config": config_name,
        "model_full": f"{MODEL_NAME}_{fn}",
        "repeat": r_i + 1, "fold": f_i + 1,
        "best_hidden": int(bp["hidden"]),
        "la_tau": LA_TAU, "mod_drop_p": MOD_DROP_P,
    })
    save_checkpoint(config_name, metrics)

    elapsed = time.time() - t0
    print(f"  [{config_name}] Fold {fold_num}/{total_folds}: "
          f"F1={metrics['f1_macro']:.4f}  AUROC={metrics['auroc_macro']:.4f}  "
          f"rec2={metrics['rec_class2']:.4f}  ({elapsed:.0f}s)", flush=True)
    return metrics

# ── メインループ（fold × 7 config を並列） ────────────────────────
t0_total = time.time()
total_folds = len(rep_obj["repeats"]) * len(rep_obj["repeats"][0]["folds"])

# 各 config のチェックポイント読み込み
done_map = {}
for cfg_name in ALL_CONFIGS:
    done_folds, _ = load_checkpoint(cfg_name)
    done_map[cfg_name] = done_folds

print(f"\n[{MODEL_NAME}] === 並列実行開始: {len(ALL_CONFIGS)} config × {total_folds} folds ===", flush=True)
print(f"[{MODEL_NAME}] ThreadPoolExecutor workers=7, DataLoader workers={N_WORKERS} per thread", flush=True)

for r_i, rep in enumerate(rep_obj["repeats"]):
    for f_i, fld in enumerate(rep["folds"]):
        fold_num = r_i * len(rep["folds"]) + f_i + 1

        # このfoldで全config完了済みならスキップ
        all_done = all((r_i+1, f_i+1) in done_map[c] for c in ALL_CONFIGS)
        if all_done:
            print(f"[{MODEL_NAME}] Fold {fold_num}/{total_folds} ALL CONFIGS DONE, SKIP", flush=True)
            continue

        print(f"\n[{MODEL_NAME}] ▶ Fold {fold_num}/{total_folds} (r={r_i+1}, f={f_i+1}) — 7 config 並列", flush=True)
        fold_t0 = time.time()

        # 7 config を ThreadPoolExecutor で並列実行
        with ThreadPoolExecutor(max_workers=7) as executor:
            futures = {
                executor.submit(
                    run_one,
                    cfg_name,
                    zf, zv, zt,
                    r_i, f_i,
                    fld["train_idx"], fld["val_idx"],
                    done_map[cfg_name],
                    total_folds
                ): cfg_name
                for cfg_name, (zf, zv, zt) in ALL_CONFIGS.items()
            }
            for future in as_completed(futures):
                cfg_name = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        done_map[cfg_name].add((r_i+1, f_i+1))
                except Exception as e:
                    print(f"  [{cfg_name}] ERROR: {e}", flush=True)
                    import traceback; traceback.print_exc()

        print(f"[{MODEL_NAME}] Fold {fold_num}/{total_folds} 完了 ({time.time()-fold_t0:.0f}s)", flush=True)

        if SMOKE:
            print(f"[{MODEL_NAME}] SMOKE: stop after 1 fold"); break
    if SMOKE: break

# ── サマリ ───────────────────────────────────────────────────────
total_time = time.time() - t0_total
print(f"\n[{MODEL_NAME}] ===== 完了 ({total_time/3600:.2f}h) =====", flush=True)
for cfg_name in ALL_CONFIGS:
    path = get_ckpt_path(cfg_name)
    if path.exists():
        df_res = pd.read_csv(path)
        df_res.to_csv(SAVE_DIR / f"{MODEL_NAME}_{cfg_name.replace('+','_')}_all25fold.csv", index=False)
        print(f"  {cfg_name}: f1_macro={df_res['f1_macro'].mean():.4f} ± {df_res['f1_macro'].std():.4f}", flush=True)
