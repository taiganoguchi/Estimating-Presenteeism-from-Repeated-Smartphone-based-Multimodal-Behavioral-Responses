"""Whisper-derived features: Q/A segments, latency, SBERT embedding, turn_id.

Ported from pipeline.ipynb cells 19, 21, 22, 23.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

# ---- SBERT (lazy) ----
_sbert_model = None
_sbert_dim = 768
_sbert_name = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"


def configure_sbert(cfg: dict) -> None:
    global _sbert_name, _sbert_dim
    sb = cfg.get("features", {}).get("sbert", {})
    _sbert_name = sb.get("model", _sbert_name)
    _sbert_dim = int(sb.get("dim", _sbert_dim))


def _load_sbert():
    global _sbert_model, _sbert_dim
    if _sbert_model is not None:
        return _sbert_model
    try:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer(_sbert_name)
        try:
            emb = _sbert_model.encode(["test"], show_progress_bar=False)
            _sbert_dim = int(emb.shape[1])
        except Exception:
            pass
        print(f"[SBERT] loaded: {_sbert_name} (dim={_sbert_dim})")
    except Exception as e:
        _sbert_model = None
        print(f"[SBERT] WARN: {e} -> hashing fallback (dim={_sbert_dim})")
    return _sbert_model


def sbert_dim() -> int:
    return _sbert_dim


def sbert_encode(text: str) -> np.ndarray:
    if not isinstance(text, str) or not text.strip():
        v = np.zeros((_sbert_dim,), dtype=np.float32)
        h = int(hashlib.blake2b(b"__EMPTY__", digest_size=8).hexdigest(), 16)
        v[h % _sbert_dim] = 1.0
        return v
    m = _load_sbert()
    if m is None:
        v = np.zeros((_sbert_dim,), dtype=np.float32)
        s = text.strip()
        grams = set()
        for n in (3, 4, 5):
            grams.update([s[i:i + n] for i in range(max(0, len(s) - n + 1))])
        if not grams:
            grams = {s}
        for g in grams:
            h = int(hashlib.blake2b(g.encode("utf-8"), digest_size=8).hexdigest(), 16)
            v[h % _sbert_dim] += 1.0
        v /= (np.linalg.norm(v) + 1e-6)
        return v
    try:
        v = m.encode([text], show_progress_bar=False)
        return np.asarray(v[0], dtype=np.float32)
    except Exception as e:
        print(f"[SBERT] encode failed: {e} -> hashing fallback")
        return sbert_encode(text + " ")


# ---- Whisper segment / latency utilities ----
def parse_whisper_segments(whisper_csv: str | Path) -> list[dict]:
    try:
        dfw = pd.read_csv(whisper_csv).sort_values(["Start", "End"], kind="mergesort")
    except Exception:
        return []
    need = {"Start", "End", "Speaker"}
    if not need.issubset(set(dfw.columns)):
        return []
    txtcol = (
        "text" if "text" in dfw.columns
        else ("Text" if "Text" in dfw.columns
              else ("Word" if "Word" in dfw.columns else None))
    )
    if txtcol is None:
        dfw["__TX__"] = ""
        txtcol = "__TX__"
    rows: list[dict] = []
    cur = None
    for _, r in dfw.iterrows():
        spk = int(r["Speaker"]); st = float(r["Start"]); en = float(r["End"])
        tx = str(r[txtcol]) if pd.notna(r[txtcol]) else ""
        if cur is None:
            cur = {"speaker": spk, "start": st, "end": en, "texts": [tx]}
        elif spk == cur["speaker"] and st <= cur["end"] + 1e-3:
            cur["end"] = max(cur["end"], en)
            cur["texts"].append(tx)
        else:
            rows.append(cur); cur = {"speaker": spk, "start": st, "end": en, "texts": [tx]}
    if cur is not None:
        rows.append(cur)
    segs: list[dict] = []
    turn = 0
    last_role = None
    for blk in rows:
        role = "Q" if blk["speaker"] == 1 else ("A" if blk["speaker"] == 2 else "U")
        if role == "Q":
            turn += 1
        elif role == "A" and last_role is None:
            turn = max(1, turn)
        text = "".join([t if isinstance(t, str) else str(t) for t in blk["texts"]]).strip()
        segs.append({
            "speaker": blk["speaker"],
            "start": float(blk["start"]),
            "end": float(blk["end"]),
            "text": text,
            "turn_id": int(turn),
            "role": role,
        })
        last_role = role
    return segs


def extract_latency_and_text(whisper_csv: str | Path) -> dict:
    try:
        df = pd.read_csv(whisper_csv)
    except Exception:
        return {"latency_q2a": np.nan, "resp_text": ""}
    if not {"Start", "End", "Speaker"}.issubset(df.columns):
        return {"latency_q2a": np.nan, "resp_text": ""}
    df = df.sort_values(["Start", "End"], kind="mergesort")
    text_col = next((c for c in ["text", "Text", "Word", "word"] if c in df.columns), None)
    q = df[df["Speaker"] == 1].copy()
    a = df[df["Speaker"] == 2].copy()
    if len(q) and len(a):
        q_end = float(q["End"].max())
        a_after = a[a["Start"] >= q_end]
        a_start = float(a_after["Start"].min()) if len(a_after) else float(a["Start"].min())
        latency = max(a_start - q_end, 0.0)
    else:
        latency = np.nan
    if len(a) and text_col is not None:
        toks = a[text_col].astype(str).tolist()
        resp_text = "".join(toks).strip()
    else:
        resp_text = ""
    return {"latency_q2a": latency, "resp_text": resp_text}


def make_turn_id_series_from_whisper(whisper_csv: str | Path, t_base: np.ndarray) -> np.ndarray:
    turn_ids = np.zeros(len(t_base), dtype=np.int32)
    try:
        df = pd.read_csv(whisper_csv)
    except Exception:
        return turn_ids
    need = {"Start", "End", "Speaker"}
    if not need.issubset(df.columns):
        return turn_ids
    df = df.sort_values(["Start", "End"], kind="mergesort")
    blocks: list[dict] = []
    cur = None
    for _, r in df.iterrows():
        spk = int(r["Speaker"]); st = float(r["Start"]); ed = float(r["End"])
        if cur is None:
            cur = {"spk": spk, "st": st, "ed": ed}
        elif spk == cur["spk"] and st <= cur["ed"] + 1e-3:
            cur["ed"] = max(cur["ed"], ed)
        else:
            blocks.append(cur); cur = {"spk": spk, "st": st, "ed": ed}
    if cur is not None:
        blocks.append(cur)
    k = 0
    for i, b in enumerate(blocks):
        if b["spk"] == 1:
            k += 1
            mask = (t_base >= b["st"]) & (t_base <= b["ed"] + 1e-9)
            turn_ids[mask] = k
            if i + 1 < len(blocks) and blocks[i + 1]["spk"] == 2:
                ba = blocks[i + 1]
                mask2 = (t_base >= ba["st"]) & (t_base <= ba["ed"] + 1e-9)
                turn_ids[mask2] = k
        elif b["spk"] == 2:
            mask = (t_base >= b["st"]) & (t_base <= b["ed"] + 1e-9)
            turn_ids[mask] = k
    return turn_ids


def build_text_and_flags_on_timeline(
    t_base: np.ndarray, segs: list[dict], sbert_encode_fn=sbert_encode, sbert_dim_v: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    if sbert_dim_v is None:
        sbert_dim_v = _sbert_dim
    T = len(t_base)
    Xt = np.zeros((T, sbert_dim_v), dtype=np.float32)
    flags = np.zeros((T, 3), dtype=np.float32)
    j = 0
    for i, t in enumerate(t_base):
        while j < len(segs) and segs[j]["end"] < t - 1e-9:
            j += 1
        k = j
        if k < len(segs) and (segs[k]["start"] - 1e-9) <= t <= (segs[k]["end"] + 1e-9):
            seg = segs[k]
            v = sbert_encode_fn(seg.get("text", ""))
            Xt[i] = v.astype(np.float32)
            flags[i, 0] = 1.0 if seg["role"] == "Q" else 0.0
            flags[i, 1] = 1.0 if seg["role"] == "A" else 0.0
            flags[i, 2] = 1.0 if len(seg.get("text", "").strip()) > 0 else 0.0
    return Xt, flags
