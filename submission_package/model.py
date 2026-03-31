"""
Multi-View Dual-Stream + Cross-View Attention 모델 (Option B)

변경사항:
  - Backbone: EfficientNet-B4 → ConvNeXt-Large (1536ch)
  - Aux Head: 영상 운동량(motion_score) 회귀 → 이미지에서 물리 운동 예측 학습
    (Knowledge Distillation from video to image model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from config import CFG


# ── Cross-View Attention ─────────────────────────────────────────────────────

class CrossViewAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn_a2b = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_b2a = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_a   = nn.LayerNorm(dim)
        self.norm_b   = nn.LayerNorm(dim)
        self.ffn_a    = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )
        self.ffn_b    = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )
        self.norm_a2  = nn.LayerNorm(dim)
        self.norm_b2  = nn.LayerNorm(dim)

    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor):
        # Cross-attention
        ctx_a, _ = self.attn_a2b(query=feat_a, key=feat_b, value=feat_b)
        feat_a   = self.norm_a(feat_a + ctx_a)
        ctx_b, _ = self.attn_b2a(query=feat_b, key=feat_a, value=feat_a)
        feat_b   = self.norm_b(feat_b + ctx_b)
        # FFN
        feat_a = self.norm_a2(feat_a + self.ffn_a(feat_a))
        feat_b = self.norm_b2(feat_b + self.ffn_b(feat_b))
        return feat_a, feat_b


# ── GeM Pooling ──────────────────────────────────────────────────────────────

class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p   = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            kernel_size=(x.size(-2), x.size(-1))
        ).pow(1.0 / self.p).flatten(1)


# ── Main Model ───────────────────────────────────────────────────────────────

class StructureModel(nn.Module):
    """
    forward(views, training=False)
      training=True  → (logits, motion_pred)  학습 시
      training=False → logits                 추론 시
    """

    def __init__(
        self,
        backbone_name:  str   = CFG.BACKBONE,
        num_classes:    int   = CFG.NUM_CLASSES,
        drop_rate:      float = CFG.DROP_RATE,
        drop_path:      float = CFG.DROP_PATH,
        video_feat_dim: int   = CFG.VIDEO_FEAT_DIM,
        pretrained:     bool  = True,
    ):
        super().__init__()

        # ── Backbone ─────────────────────────────────────────────────────
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            drop_rate=drop_rate,
            drop_path_rate=drop_path,
        )
        feat_channels = self.backbone.feature_info.channels()
        self.feat_dim = feat_channels[-1]   # ConvNeXt-Large = 1536

        # ── Projection ───────────────────────────────────────────────────
        attn_dim = 512
        self.proj = nn.Sequential(
            nn.Conv2d(self.feat_dim, attn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(attn_dim),
            nn.GELU(),
            nn.Dropout2d(p=0.1),   # Spatial Dropout 추가
        )

        # ── Cross-View Attention ─────────────────────────────────────────
        self.cross_attn = CrossViewAttention(dim=attn_dim, num_heads=8, dropout=0.15)

        # ── Pooling ──────────────────────────────────────────────────────
        self.pool = GeM(p=3.0)

        fused_dim = attn_dim * 2   # 두 뷰 concat
        self.video_feat_dim = video_feat_dim
        head_in_dim = fused_dim + video_feat_dim   # 영상 피처 직접 concat

        # ── 분류 헤드 ─────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(256, num_classes),
        )

        # ── 보조 헤드: 영상 운동량(motion_score_norm) 회귀 ─────────────────
        # 이미지만으로 구조물의 물리적 운동량을 예측하도록 강제 학습
        # → 경계(Boundary) 케이스 판별 능력 향상
        self.aux_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),   # [0, 1] 정규화된 운동량
        )

        self._init_weights()

    def _init_weights(self):
        for m in list(self.head.modules()) + list(self.aux_head.modules()):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_view(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        f = feats[-1]
        # Swin Transformer: features_only output is (B, H, W, C) → convert to (B, C, H, W)
        if f.ndim == 4 and f.shape[1] == f.shape[2]:
            f = f.permute(0, 3, 1, 2).contiguous()
        return self.proj(f)           # (B, attn_dim, H', W')

    def get_fused(self, views: torch.Tensor) -> torch.Tensor:
        B, V, C, H, W = views.shape
        x_flat  = views.view(B * V, C, H, W)
        f_flat  = self.encode_view(x_flat)
        _, D, Hf, Wf = f_flat.shape
        f_split = f_flat.view(B, V, D, Hf, Wf)

        feat_a = f_split[:, 0]
        feat_b = f_split[:, 1]

        def to_seq(t):
            return t.flatten(2).permute(0, 2, 1)   # (B, N, D)

        def to_pool(seq):
            t = seq.permute(0, 2, 1).view(B, -1, Hf, Wf)
            return self.pool(t)                     # (B, D)

        seq_a, seq_b = self.cross_attn(to_seq(feat_a), to_seq(feat_b))
        return torch.cat([to_pool(seq_a), to_pool(seq_b)], dim=1)   # (B, 2D)

    def forward(self, views: torch.Tensor, video_feats=None, training: bool = False):
        fused  = self.get_fused(views)
        if video_feats is None:
            video_feats = torch.zeros(fused.size(0), self.video_feat_dim, device=fused.device)
        head_input = torch.cat([fused, video_feats], dim=1)
        logits = self.head(head_input)
        if training:
            motion_pred = self.aux_head(fused)   # (B, 1)
            return logits, motion_pred
        return logits


# ── 빌더 ─────────────────────────────────────────────────────────────────────

def build_model(pretrained: bool = True) -> StructureModel:
    return StructureModel(pretrained=pretrained)


if __name__ == '__main__':
    model = build_model(pretrained=False)
    dummy = torch.randn(2, 2, 3, CFG.IMAGE_SIZE, CFG.IMAGE_SIZE)
    logits, motion = model(dummy, training=True)
    print(f'Input : {dummy.shape}')
    print(f'Logits: {logits.shape}')
    print(f'Motion: {motion.shape}')
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Params: {total:.1f}M')
