"""TransformerClassifier (cell 29) + factory + CrossModalTransformerClassifier (v5)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
        n_classes: int = 3,
        turn_use: bool = True,
        turn_vocab: int = 128,
        turn_emb_dim: int = 8,
        train_turn_emb: bool = False,
        use_modality_proj: bool = False,
        use_v1_modality_proj: bool = False,
        v1_text_k: int = 64,
    ):
        super().__init__()
        self.turn_use = bool(turn_use)
        self.train_turn_emb = bool(train_turn_emb)
        self.use_modality_proj = bool(use_modality_proj)
        self.use_v1_modality_proj = bool(use_v1_modality_proj)

        te = int(turn_emb_dim) if self.turn_use else 0
        if self.turn_use:
            self.turn_emb = nn.Embedding(
                num_embeddings=int(turn_vocab),
                embedding_dim=int(turn_emb_dim),
                padding_idx=0,
            )
            if not self.train_turn_emb:
                for p in self.turn_emb.parameters():
                    p.requires_grad_(False)
        else:
            self.turn_emb = None

        if self.use_modality_proj:
            # SSL-Temporal: wav2vec2(1024) + FaRL(768) + SBERT(768) = 2560
            # Symmetric 256/256/256 projection for equal treatment of modalities
            self.audio_proj = nn.Linear(1024, 256)
            self.face_proj  = nn.Linear(768,  256)
            self.text_proj  = nn.Linear(768,  256)
            self.input_ln   = nn.LayerNorm(256 * 3)
            proj_in = 256 * 3 + te
        elif self.use_v1_modality_proj:
            # Handcrafted v1: face(22) + voice(22) + SBERT(768) + flags(3) = 815
            # Symmetric projection: each modality → equal dimension k
            self.face_proj_v1  = nn.Linear(22, 22)
            self.voice_proj_v1 = nn.Linear(22, 22)
            self.text_proj_v1  = nn.Linear(768, int(v1_text_k))
            self.input_ln      = nn.LayerNorm(22 + 22 + int(v1_text_k) + 3)
            proj_in = 22 + 22 + int(v1_text_k) + 3 + te
            self.audio_proj = None; self.face_proj = None; self.text_proj = None
        else:
            self.audio_proj = None
            self.face_proj  = None
            self.text_proj  = None
            self.face_proj_v1 = None; self.voice_proj_v1 = None; self.text_proj_v1 = None
            self.input_ln   = None
            proj_in = input_dim + te

        self.proj = nn.Linear(proj_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, mask, turn_ids=None):
        B, T, D = x.shape
        if self.use_modality_proj:
            a = self.audio_proj(x[:, :, :1024])
            f = self.face_proj(x[:, :, 1024:1792])
            t_ = self.text_proj(x[:, :, 1792:2560])
            x = self.input_ln(torch.cat([a, f, t_], dim=-1))
        elif self.use_v1_modality_proj:
            f  = self.face_proj_v1 (x[:, :, 0:22])
            v  = self.voice_proj_v1(x[:, :, 22:44])
            t_ = self.text_proj_v1 (x[:, :, 44:812])
            fl = x[:, :, 812:815]
            x  = self.input_ln(torch.cat([f, v, t_, fl], dim=-1))
        if self.turn_use and turn_ids is not None:
            te = self.turn_emb(turn_ids)
            if not self.train_turn_emb:
                te = te.detach()
            x = torch.cat([x, te], dim=-1)
        x = self.proj(x)
        cls_tok = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tok, x], dim=1)
        key_padding_mask = (mask == 0)
        key_padding_mask = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=mask.device), key_padding_mask.bool()],
            dim=1,
        )
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.head(h[:, 0])


def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
    return pe


class CrossModalTransformerClassifier(nn.Module):
    """v5: Bidirectional cross-modal Transformer (MULT style, Tsai et al. ACL 2019).

    Stream α: face(22) + voice(22) = 44d time series (T frames @ 20Hz).
    Stream β: Whisper respondent-A SBERT embeddings (N_A segments per clip).

    Two independent Crossmodal Transformers:
      - crossmodal_ab: α as query, β as static memory → H_α'
      - crossmodal_ba: β as query, α as static memory → H_β'
    Each uses nn.TransformerDecoder (self-attn + cross-attn + FFN per layer).
    Masked-mean pooling produces v_α and v_β, which are concatenated with
    clip-level metadata X_meta before the classification head.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_cross_layers: int = 1,
        dropout: float = 0.3,
        n_classes: int = 3,
        alpha_dim: int = 44,   # face(22) + voice(22); flags excluded
        text_dim: int = 768,   # SBERT
        meta_dim: int = 11,    # clip-level: latency + sleep + hour_sin/cos + place_oh + state_oh
        max_frames: int = 2048,
        max_segs: int = 64,
    ):
        super().__init__()
        self.alpha_proj = nn.Linear(alpha_dim, d_model)
        self.beta_proj  = nn.Linear(text_dim, d_model)
        self.alpha_ln   = nn.LayerNorm(d_model)
        self.beta_ln    = nn.LayerNorm(d_model)
        self.register_buffer("pe", _sinusoidal_pe(max(max_frames, max_segs * 100), d_model))
        self.mod_alpha  = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mod_beta   = nn.Parameter(torch.zeros(1, 1, d_model))

        def _make_crossmodal():
            return nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=d_model, nhead=n_heads,
                    dim_feedforward=d_model * 2,
                    dropout=dropout, batch_first=True,
                ),
                num_layers=n_cross_layers,
            )

        self.crossmodal_ab = _make_crossmodal()  # α queries β
        self.crossmodal_ba = _make_crossmodal()  # β queries α

        self.head = nn.Sequential(
            nn.Linear(d_model * 2 + meta_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(
        self,
        X_alpha: torch.Tensor,       # (B, T, alpha_dim)
        mask_alpha: torch.Tensor,    # (B, T) 1=valid
        X_beta: torch.Tensor,        # (B, N_A, text_dim)
        mask_beta: torch.Tensor,     # (B, N_A) 1=valid
        t_start_beta: torch.Tensor,  # (B, N_A) segment start in seconds
        X_meta: torch.Tensor,        # (B, meta_dim)
    ) -> torch.Tensor:
        T = X_alpha.shape[1]
        pe_a = self.pe[:T].unsqueeze(0)                               # (1, T, d)
        pe_b_idx = (t_start_beta * 20).long().clamp(0, self.pe.shape[0] - 1)
        pe_b = self.pe[pe_b_idx]                                      # (B, N_A, d)

        H_a = self.alpha_ln(self.alpha_proj(X_alpha)) + pe_a + self.mod_alpha
        H_b = self.beta_ln (self.beta_proj (X_beta))  + pe_b + self.mod_beta

        kp_a = (mask_alpha == 0)  # True = masked (PyTorch key_padding_mask convention)
        kp_b = (mask_beta  == 0)

        # Bidirectional crossmodal: each stream uses the *original* other as memory (MULT style)
        H_a_out = self.crossmodal_ab(H_a, H_b,
                                      tgt_key_padding_mask=kp_a,
                                      memory_key_padding_mask=kp_b)
        H_b_out = self.crossmodal_ba(H_b, H_a,
                                      tgt_key_padding_mask=kp_b,
                                      memory_key_padding_mask=kp_a)

        m_a = mask_alpha.unsqueeze(-1).float()
        m_b = mask_beta.unsqueeze(-1).float()
        v_a = (H_a_out * m_a).sum(1) / m_a.sum(1).clamp(min=1.0)
        v_b = (H_b_out * m_b).sum(1) / m_b.sum(1).clamp(min=1.0)

        z = torch.cat([v_a, v_b, X_meta], dim=-1)
        return self.head(z)


def build_v5_model(
    d_model: int = 128,
    n_heads: int = 4,
    n_cross_layers: int = 1,
    dropout: float = 0.3,
    n_classes: int = 3,
    meta_dim: int = 11,
    device: str | None = None,
) -> CrossModalTransformerClassifier:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossModalTransformerClassifier(
        d_model=d_model, n_heads=n_heads,
        n_cross_layers=n_cross_layers,
        dropout=dropout, n_classes=n_classes,
        meta_dim=meta_dim,
    ).to(device)


class CrossModalTransformer3Stream(nn.Module):
    """v6: Canonical 3-stream MulT (Tsai et al. ACL 2019).

    Three fully independent streams: face(22d), voice(22d), text(768d).
    6 directional cross-attentions — each stream queries each other stream:
      face ← {voice, text},  voice ← {face, text},  text ← {face, voice}
    Each stream output = mean of its 2 cross-attended versions → d_model.
    Masked-mean pooling → concat [v_face, v_voice, v_text, X_meta] → MLP head.

    Ablation configs are handled by zeroing input tensors (not by masking streams):
      Full        : all streams active
      Text-only   : X_face = 0, X_voice = 0
      Audio+Text  : X_face = 0
      Face+Text   : X_voice = 0
      Audio-only  : X_face = 0,  X_text = single-zero-token
      Face-only   : X_voice = 0, X_text = single-zero-token
      Audio+Face  :              X_text = single-zero-token
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_cross_layers: int = 1,
        dropout: float = 0.3,
        n_classes: int = 3,
        face_dim: int = 22,
        voice_dim: int = 22,
        text_dim: int = 768,
        meta_dim: int = 11,
        max_frames: int = 2048,
        max_segs: int = 64,
    ):
        super().__init__()
        self.face_proj  = nn.Linear(face_dim,  d_model)
        self.voice_proj = nn.Linear(voice_dim, d_model)
        self.text_proj  = nn.Linear(text_dim,  d_model)
        self.face_ln    = nn.LayerNorm(d_model)
        self.voice_ln   = nn.LayerNorm(d_model)
        self.text_ln    = nn.LayerNorm(d_model)
        self.mod_face   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mod_voice  = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mod_text   = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_buffer("pe", _sinusoidal_pe(max(max_frames, max_segs * 100), d_model))

        def _cm():
            return nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=d_model, nhead=n_heads,
                    dim_feedforward=d_model * 2,
                    dropout=dropout, batch_first=True,
                ),
                num_layers=n_cross_layers,
            )

        # face queries {voice, text}
        self.cm_f_v = _cm()
        self.cm_f_t = _cm()
        # voice queries {face, text}
        self.cm_v_f = _cm()
        self.cm_v_t = _cm()
        # text queries {face, voice}
        self.cm_t_f = _cm()
        self.cm_t_v = _cm()

        self.head = nn.Sequential(
            nn.Linear(d_model * 3 + meta_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(
        self,
        X_face:   torch.Tensor,   # (B, T, 22)
        X_voice:  torch.Tensor,   # (B, T, 22)
        mask_fv:  torch.Tensor,   # (B, T)  1=valid, shared by face & voice
        X_text:   torch.Tensor,   # (B, N, 768)
        mask_text: torch.Tensor,  # (B, N)  1=valid
        t_start:  torch.Tensor,   # (B, N)  segment start in seconds
        X_meta:   torch.Tensor,   # (B, 11)
    ) -> torch.Tensor:
        T = X_face.shape[1]
        pe_fv = self.pe[:T].unsqueeze(0)                                        # (1, T, d)
        pe_t_idx = (t_start * 20).long().clamp(0, self.pe.shape[0] - 1)
        pe_t = self.pe[pe_t_idx]                                                # (B, N, d)

        H_f = self.face_ln (self.face_proj (X_face))  + pe_fv + self.mod_face
        H_v = self.voice_ln(self.voice_proj(X_voice)) + pe_fv + self.mod_voice
        H_t = self.text_ln (self.text_proj (X_text))  + pe_t  + self.mod_text

        kp_f = (mask_fv   == 0)   # True = padding
        kp_v = kp_f                # face & voice share the same temporal mask
        kp_t = (mask_text == 0)

        # 6 cross-attentions (each stream uses original embeddings of the other as memory)
        H_f_out = (self.cm_f_v(H_f, H_v, tgt_key_padding_mask=kp_f, memory_key_padding_mask=kp_v)
                 + self.cm_f_t(H_f, H_t, tgt_key_padding_mask=kp_f, memory_key_padding_mask=kp_t)) / 2.0
        H_v_out = (self.cm_v_f(H_v, H_f, tgt_key_padding_mask=kp_v, memory_key_padding_mask=kp_f)
                 + self.cm_v_t(H_v, H_t, tgt_key_padding_mask=kp_v, memory_key_padding_mask=kp_t)) / 2.0
        H_t_out = (self.cm_t_f(H_t, H_f, tgt_key_padding_mask=kp_t, memory_key_padding_mask=kp_f)
                 + self.cm_t_v(H_t, H_v, tgt_key_padding_mask=kp_t, memory_key_padding_mask=kp_v)) / 2.0

        m_fv = mask_fv.unsqueeze(-1).float()
        m_t  = mask_text.unsqueeze(-1).float()
        v_f = (H_f_out * m_fv).sum(1) / m_fv.sum(1).clamp(min=1.0)
        v_v = (H_v_out * m_fv).sum(1) / m_fv.sum(1).clamp(min=1.0)
        v_t = (H_t_out * m_t ).sum(1) / m_t .sum(1).clamp(min=1.0)

        z = torch.cat([v_f, v_v, v_t, X_meta], dim=-1)
        return self.head(z)


def build_v6_model(
    d_model: int = 128,
    n_heads: int = 4,
    n_cross_layers: int = 1,
    dropout: float = 0.3,
    n_classes: int = 3,
    meta_dim: int = 11,
    device: str | None = None,
) -> CrossModalTransformer3Stream:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossModalTransformer3Stream(
        d_model=d_model, n_heads=n_heads,
        n_cross_layers=n_cross_layers,
        dropout=dropout, n_classes=n_classes,
        meta_dim=meta_dim,
    ).to(device)


def build_model(
    cfg: dict,
    input_dim: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    dropout: float,
    n_classes: int = 3,
    device: str | None = None,
    use_modality_proj: bool = False,
    use_v1_modality_proj: bool = False,
    v1_text_k: int = 64,
) -> TransformerClassifier:
    tcfg = cfg.get("turn_id", {})
    model = TransformerClassifier(
        input_dim=input_dim,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
        n_classes=n_classes,
        turn_use=bool(tcfg.get("use", True)),
        turn_vocab=int(tcfg.get("num_embeddings", 128)),
        turn_emb_dim=int(tcfg.get("embed_dim", 8)),
        train_turn_emb=False,
        use_modality_proj=use_modality_proj,
        use_v1_modality_proj=use_v1_modality_proj,
        v1_text_k=v1_text_k,
    )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device)
