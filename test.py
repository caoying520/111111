# Model.py
# 三模态全局特征融合模型：适用于 ESM2(1280)、UniRep(1024)、ProtT5(1900) 的 pooled vector
# 输入形状分别为 [B, 1280]、[B, 1024]、[B, 1900]，不是残基级 [B, L, D]。

import torch
import torch.nn as nn


class ModalityProjector(nn.Module):
    """每种预训练模型独立投影到同一维度。"""
    def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class GlobalFeatureFusion(nn.Module):
    """
    将三种全局向量看成三个“模态 token”，只在 ESM2 / UniRep / ProtT5
    三个模态之间做注意力，不把特征维度伪装成氨基酸位置。
    """
    def __init__(
        self,
        esm_dim=1280,
        unirep_dim=1024,
        prott5_dim=1900,
        hidden_dim=256,
        num_heads=4,
        num_layers=1,
        dropout=0.2,
        num_classes=3,
    ):
        super().__init__()

        self.esm_proj = ModalityProjector(esm_dim, hidden_dim, dropout)
        self.uni_proj = ModalityProjector(unirep_dim, hidden_dim, dropout)
        self.proj_proj = ModalityProjector(prott5_dim, hidden_dim, dropout)

        # 可学习的模态身份编码：区分三种特征来源
        self.modality_embedding = nn.Parameter(torch.zeros(1, 3, hidden_dim))
        nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),
        )

        # 学习三个模态 token 的融合权重
        self.pool_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_esm, x_unirep, x_prott5, return_attention=False):
        # 每个输入必须是一个样本一个全局向量：[B, D]
        if x_esm.ndim != 2 or x_unirep.ndim != 2 or x_prott5.ndim != 2:
            raise ValueError(
                "本模型接收全局 pooled features [B,D]；"
                f"实际输入形状为 {x_esm.shape}, {x_unirep.shape}, {x_prott5.shape}"
            )

        z_esm = self.esm_proj(x_esm)
        z_uni = self.uni_proj(x_unirep)
        z_pro = self.proj_proj(x_prott5)

        # [B, 3, hidden_dim]：三个模态 token
        tokens = torch.stack([z_esm, z_uni, z_pro], dim=1)
        tokens = tokens + self.modality_embedding
        fused_tokens = self.fusion_encoder(tokens)

        weights = torch.softmax(self.pool_score(fused_tokens).squeeze(-1), dim=1)
        fused = torch.sum(fused_tokens * weights.unsqueeze(-1), dim=1)

        logits = self.classifier(fused)
        if return_attention:
            return logits, weights
        return logits


# 保留旧训练代码中 ProtSATT 的类名，便于理解；它已不是原始 ProtSATT 结构。
class ProtSATT(GlobalFeatureFusion):
    pass
