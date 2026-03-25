# Selective State Space Model (SSM) implementation based on:
#
#   Mamba  — "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
#             Albert Gu*, Tri Dao*  (https://arxiv.org/abs/2312.00752)
#   Mamba2 — "Transformers are SSMs: Generalized Models and Efficient Algorithms
#             Through Structured State Space Duality"
#             Tri Dao*, Albert Gu*  (https://arxiv.org/abs/2405.21060)
#
# This file contains an independent pure-PyTorch implementation of the algorithm
# described in the above papers. No code from the mamba-ssm library is copied here.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch AMP (Automatic Mixed Precision) 가속화 설정
# Keras의 mixed_precision 대신 사용. accelerate launch 실행 권장.
scaler = torch.cuda.amp.GradScaler()

from .config import CROP_LEN, NUM_CLASSES, UMAP_OUTPUT_DIM


# ──────────────────────────────────────────────────────────────────────────────
# Selective SSM (Mamba-style, pure PyTorch)
# ──────────────────────────────────────────────────────────────────────────────

class SelectiveSSM(nn.Module):
    """
    Selective State Space Model — Mamba 논문의 핵심 연산을 순수 PyTorch로 구현.

    핵심 재귀식:
        h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * x_t
        y_t = C_t * h_t + D * x_t

    A, B, C, dt 는 모두 입력 x에 의존 (selective).

    Args:
        d_model:  입력/출력 차원
        d_state:  SSM hidden state 차원 N
        d_conv:   입력 causal conv 커널 크기
        expand:   내부 채널 확장 비율 (d_inner = expand * d_model)
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)

        # x, z 두 브랜치로 분리
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 인과적 depthwise conv (conv 이후 SiLU 활성화)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,   # 나중에 미래 부분 잘라냄
            groups=self.d_inner,
            bias=True,
        )

        # dt, B, C 선형 투영
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A: (d_inner, d_state) — 로그 스케일로 저장
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D: skip connection 계수
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model)
        """
        B, L, _ = x.shape

        xz = self.in_proj(x)                         # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)               # 각각 (B, L, d_inner)

        # Causal conv: (B, d_inner, L) → 미래 L개 패딩을 잘라냄
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)                      # (B, L, d_inner)

        # SSM 파라미터 계산
        x_dbl = self.x_proj(x_conv)                  # (B, L, dt_rank + 2*d_state)
        dt, B_mat, C_mat = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))            # (B, L, d_inner)

        A = -torch.exp(self.A_log.float())           # (d_inner, d_state)

        # 순차적 scan — dA/dB를 타임스텝마다 계산 (메모리 효율적)
        # [주의] 전체를 미리 계산하면 (B, L, d_inner, d_state) 텐서가 생성됨.
        # 예: B=32, L=281, d_inner=128, d_state=128 → dA+dB 합산 약 1.1 GB → 메모리 부족.
        h = x_conv.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for i in range(L):
            dA_i = torch.exp(dt[:, i].unsqueeze(-1) * A)              # (B, d_inner, d_state)
            dB_i = dt[:, i].unsqueeze(-1) * B_mat[:, i].unsqueeze(1)  # (B, d_inner, d_state)
            h = dA_i * h + dB_i * x_conv[:, i].unsqueeze(-1)
            y_i = (h * C_mat[:, i].unsqueeze(1)).sum(-1)               # (B, d_inner)
            ys.append(y_i)

        y = torch.stack(ys, dim=1)                   # (B, L, d_inner)
        y = y + x_conv * self.D                      # skip connection

        # 게이팅
        out = y * F.silu(z)
        return self.out_proj(out)                    # (B, L, d_model)


# ──────────────────────────────────────────────────────────────────────────────
# CNN 구성 요소 (기존 Keras backbone의 PyTorch 재구현)
# ──────────────────────────────────────────────────────────────────────────────

class CausalDWConv1d(nn.Module):
    """인과적 Depthwise Conv1D (미래 정보 누출 방지)."""

    def __init__(self, channels: int, kernel_size: int = 17, dilation_rate: int = 1):
        super().__init__()
        self.padding = dilation_rate * (kernel_size - 1)
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            dilation=dilation_rate,
            padding=0,
            groups=channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = x.transpose(1, 2)
        x = F.pad(x, (self.padding, 0))
        x = self.conv(x)
        return x.transpose(1, 2)


class ECA(nn.Module):
    """Efficient Channel Attention."""

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        gap = x.mean(dim=1)           # (B, C)
        attn = self.conv(gap.unsqueeze(1)).squeeze(1)  # (B, C)
        return x * torch.sigmoid(attn).unsqueeze(1)


class Conv1DBlock(nn.Module):
    """
    Efficient Conv1D Block (기존 Keras Conv1DBlock의 PyTorch 재구현).
    Linear expand → CausalDWConv1d → BN → ECA → Linear project → residual
    """

    def __init__(
        self,
        channel_size: int,
        kernel_size: int = 17,
        dilation_rate: int = 1,
        drop_rate: float = 0.0,
        expand_ratio: int = 2,
    ):
        super().__init__()
        channels_expand = channel_size * expand_ratio

        self.expand  = nn.Linear(channel_size, channels_expand, bias=True)
        self.act     = nn.SiLU()
        self.dw_conv = CausalDWConv1d(channels_expand, kernel_size, dilation_rate)
        self.bn      = nn.BatchNorm1d(channels_expand)
        self.eca     = ECA()
        self.project = nn.Linear(channels_expand, channel_size, bias=True)
        self.drop    = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.act(self.expand(x))
        x = self.dw_conv(x)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.eca(x)
        x = self.drop(self.project(x))
        return x + skip


class GatedMLP(nn.Module):
    """Gated MLP (FFN)."""

    def __init__(self, d_model: int, expand: int = 2):
        super().__init__()
        hidden = d_model * expand
        self.fc1 = nn.Linear(d_model, hidden * 2, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(y * F.silu(gate))


# ──────────────────────────────────────────────────────────────────────────────
# Mamba 블록 & 전체 모델
# ──────────────────────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """
    Mamba 블록 (기존 TransformerBlock 대체).
    LayerNorm → SelectiveSSM → residual
    LayerNorm → GatedMLP     → residual
    """

    def __init__(self, d_model: int, d_state: int = 16, drop_rate: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mixer = SelectiveSSM(d_model, d_state=d_state)
        self.mlp   = GatedMLP(d_model, expand=2)
        self.drop  = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.mixer(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class SignMamba(nn.Module):
    """
    CNN + Mamba 수어 인식 모델.

    기존 Transformer 모델과 대칭 구조:
      Conv1DBlock × 3 + TransformerBlock  →  Conv1DBlock × 3 + MambaBlock

    Args:
        max_len:     입력 시퀀스 최대 길이 (default: 125)
        in_dim:      UMAP 출력 차원 (default: 32)
        d_model:     모델 내부 차원
        n_stages:    (Conv×3 + Mamba) 스테이지 수 (default: 2)
        kernel_size: Conv1DBlock 커널 크기 (default: 17)
        d_state:     SSM hidden state 차원 (default: 16)
        drop_rate:   Conv dropout (default: 0.2)
        clf_dropout: Classifier dropout (default: 0.8)
        num_classes: 분류 클래스 수 (default: 196)
    """

    def __init__(
        self,
        max_len: int = CROP_LEN,
        in_dim: int = UMAP_OUTPUT_DIM,
        d_model: int = 32,
        n_stages: int = 2,
        kernel_size: int = 17,
        d_state: int = 16,
        drop_rate: float = 0.2,
        clf_dropout: float = 0.8,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()

        # Stem
        self.stem_linear = nn.Linear(in_dim, d_model, bias=False)
        self.stem_bn     = nn.BatchNorm1d(d_model)

        # Stages: n_stages × (Conv1DBlock×3 + MambaBlock)
        stages = []
        for _ in range(n_stages):
            for _ in range(3):
                stages.append(Conv1DBlock(d_model, kernel_size, drop_rate=drop_rate))
            stages.append(MambaBlock(d_model, d_state=d_state, drop_rate=drop_rate))
        self.stages = nn.Sequential(*stages)

        # Top
        self.top_linear = nn.Linear(d_model, d_model * 2)
        self.clf_drop   = nn.Dropout(clf_dropout)
        self.classifier = nn.Linear(d_model * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, in_dim)
        Returns:
            logits: (B, num_classes)
        """
        x = self.stem_linear(x)
        x = self.stem_bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.stages(x)
        x = self.top_linear(x).mean(dim=1)   # global avg pool
        x = self.clf_drop(x)
        return self.classifier(x)


def get_mamba_model(
    max_len: int = CROP_LEN,
    in_dim: int = UMAP_OUTPUT_DIM,
    d_model: int = 32,
    n_stages: int = 2,
    num_classes: int = NUM_CLASSES,
    **kwargs,
) -> SignMamba:
    """SignMamba 모델 생성 헬퍼 (get_model()과 대칭)."""
    return SignMamba(
        max_len=max_len,
        in_dim=in_dim,
        d_model=d_model,
        n_stages=n_stages,
        num_classes=num_classes,
        **kwargs,
    )
