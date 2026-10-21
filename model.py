"""DEMON-guided Mel/MATS fusion model.

Input
-----
x: torch.Tensor of shape (batch, 3, frequency_bins, time_frames).
The channels must be ordered as Mel band energy, DEMON modulation
spectrogram, and MATS features. Mel and MATS must already be
aligned to shared acoustic bands by preprocessing.

Output
------
DemonGatedFusionModel returns (features, logits, demon_logits):
    features:     (batch, feat_dim), for the main classifier / center loss.
    logits:       (batch, n_classes), the main classification scores.
    demon_logits: (batch, n_classes), the auxiliary DEMON scores.
These outputs are returned in both training and evaluation modes.

Initialization
--------------
The acoustic and classification heads are created on the first forward
pass because their dimensions depend on the input geometry. Run a dummy
forward with the intended feature shape before creating an optimizer or
loading a trained state_dict. Keep that feature geometry fixed afterward.
For inference, initialize the heads, load weights, and then call eval().

Dependencies: torch, einops.
CenterLoss is retained for compatibility with the original training loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

__all__ = ["DemonGatedFusionModel", "CenterLoss"]



class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class BranchEncoderHybrid(nn.Module):
    """
    Hybrid branch encoder: temporal + frequency + local TF convolution.

    Input:  (B, 1, F, T)
    Output: (B, c_out, F, T/4)

    Steps:
        1. 1×11 temporal conv
        2. 5×1 depthwise frequency conv
        3. 3×3 2-D TF conv
        4. SEBlock
        5. 1×4 temporal average pooling
        6. Dropout
    """
    def __init__(self, c_out: int = 32, drop_p: float = 0.3):
        super().__init__()

        # 1. Temporal conv
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, c_out, kernel_size=(1, 11), padding=(0, 5), bias=False),
            nn.BatchNorm2d(c_out),
            nn.ELU(inplace=True),
        )

        # 2. Frequency depthwise conv
        self.freq_conv = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=(5, 1), padding=(2, 0),
                      groups=c_out, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ELU(inplace=True),
        )

        # 3. 2-D TF conv
        self.tf_conv = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(c_out),
            nn.ELU(inplace=True),
        )

        # 4. SE
        self.se = SEBlock(c_out)

        # 5. Temporal pooling only (keep frequency axis intact)
        self.pool = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))

        # 6. Dropout
        self.drop = nn.Dropout(drop_p)

    def forward(self, x):
        # x: (B, 1, F, T)
        x = self.temporal_conv(x)   # (B, c_out, F, T)
        x = self.freq_conv(x)       # (B, c_out, F, T)
        x = self.tf_conv(x)         # (B, c_out, F, T)
        x = self.se(x)              # (B, c_out, F, T)
        x = self.pool(x)            # (B, c_out, F, T/4)
        x = self.drop(x)
        return x


class MultiScaleFusion2D(nn.Module):
    """
    Multi-scale 2D fusion after channel-wise Mel/MATS concatenation.

    Input:  (B, 2*branch_c, F, T/4)
    Output: (B, branch_c,   F, T/4)

    Steps:
        1. 1×1 conv: 2*branch_c -> branch_c  (reduce)
        2. Three parallel convolutions (3×3, 5×5, 1×7)
        3. Cat -> 3*branch_c
        4. 1×1 conv: 3*branch_c -> branch_c  (fuse)
        5. Residual add with step-1 output
    """
    def __init__(self, c_in: int, c_out: int, drop_p: float = 0.3):
        super().__init__()
        assert c_in == 2 * c_out, (
            f"MultiScaleFusion2D expects c_in == 2*c_out, got {c_in} vs {c_out}"
        )

        # Step 1: dimension reduction
        self.reduce = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ELU(inplace=True),
        )

        # Step 2: parallel multi-scale branches
        self.branch_3x3 = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(c_out),
            nn.ELU(inplace=True),
        )
        self.branch_5x5 = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=(5, 5), padding=(2, 2), bias=False),
            nn.BatchNorm2d(c_out),
            nn.ELU(inplace=True),
        )
        self.branch_1x7 = nn.Sequential(
            nn.Conv2d(c_out, c_out, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(c_out),
            nn.ELU(inplace=True),
        )

        # Step 4: fuse 3 branches back to c_out
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * c_out, c_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ELU(inplace=True),
            nn.Dropout(drop_p),
        )

    def forward(self, x):
        # x: (B, 2*branch_c, F, T/4)
        reduced = self.reduce(x)                            # (B, branch_c, F, T/4)

        b3 = self.branch_3x3(reduced)                      # (B, branch_c, F, T/4)
        b5 = self.branch_5x5(reduced)                      # (B, branch_c, F, T/4)
        bt = self.branch_1x7(reduced)                      # (B, branch_c, F, T/4)

        multi = torch.cat([b3, b5, bt], dim=1)             # (B, 3*branch_c, F, T/4)
        out = self.fuse(multi)                              # (B, branch_c, F, T/4)

        # Residual connection
        out = out + reduced                                 # (B, branch_c, F, T/4)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        assert emb_size % num_heads == 0
        self.num_heads = num_heads
        self.scale = (emb_size // num_heads) ** 0.5
        self.qkv = nn.Linear(emb_size, 3 * emb_size)
        self.proj = nn.Linear(emb_size, emb_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        att = (q @ k.transpose(-2, -1)) / self.scale
        att = self.drop(F.softmax(att, dim=-1))
        out = (att @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, emb_size: int, num_heads: int = 8, ff_mult: int = 4, drop_p: float = 0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = MultiHeadAttention(emb_size, num_heads, drop_p)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * ff_mult),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(emb_size * ff_mult, emb_size),
            nn.Dropout(drop_p),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class FusedPatchProjection(nn.Module):
    def __init__(self, c_in: int, emb_size: int, patch_h: int = 8, patch_w: int = 4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(c_in, emb_size,
                      kernel_size=(patch_h, patch_w),
                      stride=(patch_h, patch_w),
                      bias=False),
            nn.BatchNorm2d(emb_size),
        )
        self.flatten = Rearrange("b e h w -> b (h w) e")

    def forward(self, x):
        return self.flatten(self.proj(x))


class AcousticHead(nn.Module):
    def __init__(self, token_dim: int, feat_dim: int = 32, drop_p: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, 256),
            nn.ELU(inplace=True),
            nn.Dropout(drop_p),
            nn.Linear(256, feat_dim),
            nn.ELU(inplace=True),
            nn.Dropout(drop_p * 0.5),
        )

    def forward(self, x):
        return self.net(x.flatten(1))


class DemonEncoder(nn.Module):
    """Independent DEMON encoder for channel gating."""
    def __init__(self, c_out: int = 32, feat_dim: int = 32, drop_p: float = 0.3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, c_out, kernel_size=(5, 9), padding=(2, 4), bias=False),
            nn.BatchNorm2d(c_out),
            nn.ELU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.BatchNorm2d(c_out),
            nn.ELU(inplace=True),
            SEBlock(c_out),
            nn.AvgPool2d(kernel_size=(2, 4), stride=(2, 4)),
            nn.Dropout(drop_p),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_out * 4 * 4, 128),
            nn.ELU(inplace=True),
            nn.Dropout(drop_p),
            nn.Linear(128, feat_dim),
            nn.ELU(inplace=True),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return self.head(x)


class DemonGuidedChannelGate(nn.Module):
    """Use DEMON features to softly reweight acoustic feature-map channels."""
    def __init__(self, demon_dim: int, channels: int, init_scale: float = 0.1):
        super().__init__()
        hidden = max(demon_dim, channels)
        self.net = nn.Sequential(
            nn.Linear(demon_dim, hidden),
            nn.ELU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, acoustic_map, demon_feat):
        gate = self.net(demon_feat).unsqueeze(-1).unsqueeze(-1)
        return acoustic_map * (1.0 + self.alpha * gate)


class CenterLoss(nn.Module):
    def __init__(self, num_classes: int = 4, feat_dim: int = 32):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, x, labels):
        B = x.size(0)
        dist = (
            x.pow(2).sum(1, keepdim=True).expand(B, self.centers.size(0))
            + self.centers.pow(2).sum(1).expand(B, self.centers.size(0))
            - 2 * x @ self.centers.t()
        )
        mask = torch.zeros_like(dist).scatter_(1, labels.unsqueeze(1), 1)
        loss = (dist * mask).clamp(min=1e-12).sum() / B
        return loss


class _DemonGatedModelBase(nn.Module):
    """
    Shared Mel-MATS fusion and DEMON-guided channel-gating backbone.

    Mel  -> BranchEncoderHybrid -> mel_feat  (B, branch_c, F, T/4)
    MATS-> BranchEncoderHybrid -> mats_feat(B, branch_c, F, T/4)
    cat  -> (B, 2*branch_c, F, T/4)
    MultiScaleFusion2D -> (B, branch_c, F, T/4)
    DEMON -> DemonEncoder -> channel gate for fused acoustic map
    FusedPatchProjection -> Transformer -> AcousticHead -> acoustic_feat
    """
    def __init__(
        self,
        n_freqs: int = 128,
        branch_c: int = 32,
        emb_size: int = 64,
        depth: int = 4,
        n_classes: int = 4,
        drop_p: float = 0.4,
        feat_dim: int = 32,
        patch_h: int = 8,
        patch_w: int = 4,
    ):
        super().__init__()
        self.n_freqs = n_freqs

        # Mel and MATS encoders — same structure, separate parameters
        self.mel_encoder = BranchEncoderHybrid(c_out=branch_c, drop_p=drop_p * 0.6)
        # Legacy attribute name retained for checkpoint compatibility; encodes MATS.
        self.lofar_encoder = BranchEncoderHybrid(c_out=branch_c, drop_p=drop_p * 0.6)

        # Multi-scale feature fusion
        self.feature_fuser = MultiScaleFusion2D(
            c_in=2 * branch_c,
            c_out=branch_c,
            drop_p=drop_p * 0.6,
        )

        # Patch projection + Transformer
        self.patch_proj = FusedPatchProjection(
            c_in=branch_c,
            emb_size=emb_size,
            patch_h=patch_h,
            patch_w=patch_w,
        )
        self.transformer = nn.Sequential(
            *[TransformerBlock(emb_size,
                               num_heads=max(emb_size // 8, 1),
                               drop_p=drop_p)
              for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(emb_size)

        # DEMON encoder, channel gate, and auxiliary classifier
        self.demon_encoder = DemonEncoder(
            c_out=branch_c, feat_dim=feat_dim, drop_p=drop_p * 0.6
        )
        self.demon_gate = DemonGuidedChannelGate(
            demon_dim=feat_dim,
            channels=branch_c,
            init_scale=0.1,
        )
        self.demon_aux_classifier = nn.Linear(feat_dim, n_classes)

        # Store for deferred head construction
        self._emb_size = emb_size
        self._feat_dim = feat_dim
        self._n_classes = n_classes
        self._drop_p = drop_p
        self._head_built = False

    # ── deferred head construction (token count depends on T) ──────────────
    def _build_heads(self, n_tokens: int):
        raise NotImplementedError

    # ── acoustic stream ────────────────────────────────────────────────────
    def encode_acoustic(self, x, demon_feat=None):
        mel = x[:, 0:1]                                 # (B, 1, F, T)
        mats = x[:, 2:3]                               # (B, 1, F, T)

        mel_feat = self.mel_encoder(mel)                # (B, branch_c, F, T/4)
        mats_feat = self.lofar_encoder(mats)          # (B, branch_c, F, T/4)

        fused = torch.cat([mel_feat, mats_feat], dim=1)  # (B, 2*branch_c, F, T/4)
        fused = self.feature_fuser(fused)               # (B, branch_c, F, T/4)
        if demon_feat is not None:
            fused = self.demon_gate(fused, demon_feat)  # (B, branch_c, F, T/4)

        tokens = self.patch_proj(fused)                 # (B, N, emb_size)
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        if not self._head_built:
            self._build_heads(tokens.shape[1])

        acoustic_feat = self.acoustic_head(tokens)      # (B, feat_dim)
        return acoustic_feat

    # ── DEMON stream ───────────────────────────────────────────────────────
    def encode_demon(self, x):
        demon = x[:, 1:2]                               # (B, 1, F, T)
        return self.demon_encoder(demon)


class DemonGatedFusionModel(_DemonGatedModelBase):
    """
    Main model: Mel-MATS feature fusion + DEMON-guided channel gating.

    DEMON features modulate the fused acoustic feature map, but are not
    concatenated with the final acoustic feature before classification.

    forward returns:
        (feat, logits, demon_logits)
    """
    def _build_heads(self, n_tokens: int):
        device = next(self.parameters()).device
        self.acoustic_head = AcousticHead(
            token_dim=n_tokens * self._emb_size,
            feat_dim=self._feat_dim,
            drop_p=self._drop_p,
        ).to(device)
        self.classifier = nn.Linear(self._feat_dim, self._n_classes).to(device)
        self._head_built = True

    def forward(self, x):
        demon_feat = self.encode_demon(x)
        demon_logits = self.demon_aux_classifier(demon_feat)
        feat = self.encode_acoustic(x, demon_feat=demon_feat)
        logits = self.classifier(feat)
        return feat, logits, demon_logits

if __name__ == "__main__":
    torch.manual_seed(0)
    model = DemonGatedFusionModel(n_classes=4)
    x = torch.randn(2, 3, 128, 156)
    with torch.no_grad():
        model(x)  # Initialize the input-dependent heads.
        model.eval()
        features, logits, demon_logits = model(x)
    print(f"features: {tuple(features.shape)}")
    print(f"logits: {tuple(logits.shape)}")
    print(f"demon_logits: {tuple(demon_logits.shape)}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
