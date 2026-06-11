"""All distillation loss functions for Weeks 3–5."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Week 3: Response-Based KD (Hinton et al., 2015) ──────────────────────────

class HintonKDLoss(nn.Module):
    """
    L = alpha * T^2 * KL(teacher_soft || student_soft) + (1-alpha) * CE
    T^2 compensates for gradient shrinkage when dividing logits by T.
    """
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.T = temperature
        self.alpha = alpha

    def forward(self, s_logits, t_logits, labels):
        soft_t = F.softmax(t_logits / self.T, dim=1)
        log_soft_s = F.log_softmax(s_logits / self.T, dim=1)
        L_kd = self.T ** 2 * F.kl_div(log_soft_s, soft_t, reduction="batchmean")
        L_ce = F.cross_entropy(s_logits, labels)
        return self.alpha * L_kd + (1 - self.alpha) * L_ce


# ── Week 4: Feature-Based KD (FitNets) ───────────────────────────────────────

class FeatureKDLoss(nn.Module):
    """
    Adapts student feature channels to match teacher via 1x1 conv, then MSE.
    L = beta * MSE(adapter(F_s), F_t) + (1-beta) * CE
    """
    def __init__(self, student_channels, teacher_channels, beta=0.3):
        super().__init__()
        self.beta = beta
        # 1x1 conv + BN to project student channels to teacher channels
        self.adapter = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(teacher_channels),
        )

    def forward(self, s_logits, t_logits, labels, s_feat, t_feat):
        projected = self.adapter(s_feat)
        L_feat = F.mse_loss(projected, t_feat.detach())
        L_ce = F.cross_entropy(s_logits, labels)
        return self.beta * L_feat + (1 - self.beta) * L_ce


# ── Week 4: Attention Transfer (Zagoruyko & Komodakis, 2017) ─────────────────

class AttentionTransferLoss(nn.Module):
    """
    Match spatial attention maps: A(F) = sum_c |F_c|^2, then L2-normalise.
    L = beta * ||A_norm(F_t) - A_norm(F_s)||^2 + (1-beta) * CE
    """
    def __init__(self, beta=0.3):
        super().__init__()
        self.beta = beta

    def _attention(self, feat):
        # feat: [B, C, H, W] -> [B, H*W] normalised
        a = feat.pow(2).sum(dim=1)          # [B, H, W]
        a = a.view(a.size(0), -1)           # [B, H*W]
        return F.normalize(a, p=2, dim=1)

    def forward(self, s_logits, t_logits, labels, s_feat, t_feat):
        L_at = F.mse_loss(self._attention(s_feat), self._attention(t_feat.detach()))
        L_ce = F.cross_entropy(s_logits, labels)
        return self.beta * L_at + (1 - self.beta) * L_ce


# ── Week 5: Relation-Based KD (Park et al., CVPR 2019) ───────────────────────

class RKDLoss(nn.Module):
    """
    Distance-wise: Huber loss on normalised pairwise distances.
    Angle-wise:    Huber loss on cosine of triplet angles.
    L = lambda_d * L_D + lambda_a * L_A + CE
    """
    def __init__(self, lambda_d=25.0, lambda_a=50.0):
        super().__init__()
        self.lambda_d = lambda_d
        self.lambda_a = lambda_a

    def _pdist(self, e):
        diff = e.unsqueeze(0) - e.unsqueeze(1)           # [N, N, d]
        d = diff.pow(2).sum(-1).clamp(min=1e-12).sqrt()  # [N, N]
        return d / (d.mean() + 1e-8)

    def forward(self, s_logits, t_logits, labels, s_emb, t_emb):
        t_emb = t_emb.detach()

        # Distance-wise loss
        L_d = F.huber_loss(self._pdist(s_emb), self._pdist(t_emb))

        # Angle-wise loss
        td = t_emb.unsqueeze(0) - t_emb.unsqueeze(1)    # [N, N, d]
        sd = s_emb.unsqueeze(0) - s_emb.unsqueeze(1)
        cos_t = F.cosine_similarity(td.unsqueeze(1), td.unsqueeze(0), dim=-1)
        cos_s = F.cosine_similarity(sd.unsqueeze(1), sd.unsqueeze(0), dim=-1)
        L_a = F.huber_loss(cos_s, cos_t.detach())

        L_ce = F.cross_entropy(s_logits, labels)
        return self.lambda_d * L_d + self.lambda_a * L_a + L_ce


# ── Week 5: Combined KD + Feature ────────────────────────────────────────────

class CombinedKDLoss(nn.Module):
    """
    L = alpha * T^2 * KL + beta * MSE(adapter(F_s), F_t) + (1-alpha-beta) * CE
    """
    def __init__(self, student_channels, teacher_channels,
                 temperature=4.0, alpha=0.5, beta=0.3):
        super().__init__()
        assert alpha + beta < 1.0, (
            f"alpha+beta must be < 1 to leave weight for the CE term, "
            f"got alpha={alpha}, beta={beta}"
        )
        self.T = temperature
        self.alpha = alpha
        self.beta = beta
        self.adapter = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(teacher_channels),
        )

    def forward(self, s_logits, t_logits, labels, s_feat, t_feat):
        soft_t = F.softmax(t_logits / self.T, dim=1)
        log_soft_s = F.log_softmax(s_logits / self.T, dim=1)
        L_kd = self.T ** 2 * F.kl_div(log_soft_s, soft_t, reduction="batchmean")
        L_feat = F.mse_loss(self.adapter(s_feat), t_feat.detach())
        L_ce = F.cross_entropy(s_logits, labels)
        return self.alpha * L_kd + self.beta * L_feat + (1 - self.alpha - self.beta) * L_ce
