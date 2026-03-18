"""
mamba_train.py — CNN + Mamba2 수어 인식 모델 학습 스크립트 (PyTorch)

기존 gloss_transformer_train.py와 대칭 구조.
데이터 전처리(UMAP 포함)는 기존 DataSetter(TF)를 그대로 재사용하고,
TF Dataset → numpy → PyTorch DataLoader 브릿지로 연결.

실행 예시:
    python -m train.mamba_train --epochs 300 --bs 32 --lr 0.0001 --d_model 32
"""

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wandb

from src.mamba_backbone import get_mamba_model
from src.config import (
    CROP_LEN, LEARNING_RATE, EPOCHS, BATCH_SIZE, NUM_CLASSES,
    VALIDATION_SPLIT, WEIGHT_DECAY, UMAP_OUTPUT_DIM, UMAP_LOAD_PATH,
    LOCAL_PATHS, WANDB_GM_PROJECT, WANDB_GM_GROUP, WANDB_GM_TAGS,
    L_GM, L_CKPT, date_idx,
)
from load_data.create_dataset import DataSetter, upload_file


# ──────────────────────────────────────────────
# TF Dataset → PyTorch Dataset 브릿지
# ──────────────────────────────────────────────

class TFDatasetBridge(Dataset):
    """
    기존 DataSetter가 반환하는 tf.data.Dataset을 PyTorch Dataset으로 변환.
    TF Dataset을 unbatch → numpy iterator로 읽어 메모리에 적재.
    """

    def __init__(self, tf_dataset, max_seq_len: int = CROP_LEN):
        print("  TF Dataset → numpy 변환 중...")
        sequences, labels = [], []
        for x, y in tf_dataset.unbatch().as_numpy_iterator():
            sequences.append(x[:max_seq_len])
            labels.append(int(y))
        self.sequences = np.array(sequences, dtype=np.float32)  # (N, T, D)
        self.labels = np.array(labels, dtype=np.int64)          # (N,)
        print(f"  변환 완료: {len(self.labels)}개 샘플, shape={self.sequences.shape}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.sequences[idx])   # (T, D)
        y = torch.tensor(self.labels[idx])           # scalar
        return x, y


# ──────────────────────────────────────────────
# 학습 루프
# ──────────────────────────────────────────────

def train_model(
    learning_rate: float = LEARNING_RATE,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    weight_decay: float = WEIGHT_DECAY,
    max_sequence_len: int = CROP_LEN,
    d_model: int = UMAP_OUTPUT_DIM,
    n_stages: int = 2,
    d_state: int = 16,
):
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"사용 디바이스: {device}")

    # W&B 초기화
    run = wandb.init(
        project=WANDB_GM_PROJECT,
        name=f"mamba-d{d_model}-s{n_stages}",
        group=WANDB_GM_GROUP,
        tags=list(WANDB_GM_TAGS) + ["mamba"],
        job_type="train",
        config={
            "model_type": "CNN+Mamba",
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "max_len": max_sequence_len,
            "d_model": d_model,
            "n_stages": n_stages,
            "d_state": d_state,
            "num_classes": NUM_CLASSES,
            "input_dim": UMAP_OUTPUT_DIM,
        },
    )

    # ── 1. 데이터 로드 (기존 DataSetter 재사용) ──
    print("\n데이터 로딩 중 (DataSetter + UMAP)...")
    train_tf, val_tf, test_tf = DataSetter(
        umap_path=UMAP_LOAD_PATH,
        max_seq_len=max_sequence_len,
        batch_size=batch_size,
        dim_reduction=True,
    ).get_datasets()

    print("\nTF Dataset → PyTorch DataLoader 변환 중...")
    train_ds = TFDatasetBridge(train_tf, max_sequence_len)
    val_ds   = TFDatasetBridge(val_tf,   max_sequence_len)
    test_ds  = TFDatasetBridge(test_tf,  max_sequence_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # ── 2. 모델 생성 ──
    print("\n모델 생성 중...")
    model = get_mamba_model(
        max_len=max_sequence_len,
        in_dim=UMAP_OUTPUT_DIM,
        d_model=d_model,
        n_stages=n_stages,
        d_state=d_state,
        num_classes=NUM_CLASSES,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"파라미터 수: {n_params:,}")
    wandb.config.update({"n_params": n_params})

    # ── 3. 손실함수 / 옵티마이저 / 스케줄러 ──
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    # ── 4. 학습 루프 ──
    ckpt_path  = Path(L_CKPT) / f"gloss_mamba_{date_idx}.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 10

    print("\n학습 시작...")
    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total += len(y)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Validate ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
                val_correct += (logits.argmax(dim=1) == y).sum().item()
                val_total += len(y)

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={current_lr:.2e}"
        )

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "lr": current_lr,
        })

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → 체크포인트 저장 (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping (patience={early_stop_patience})")
                break

    # ── 5. 최적 가중치 로드 후 평가 ──
    print("\n최적 모델로 테스트 평가 중...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    test_loss, test_correct, test_total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            test_loss += loss.item() * len(y)
            test_correct += (logits.argmax(dim=1) == y).sum().item()
            test_total += len(y)

    test_loss /= test_total
    test_acc = test_correct / test_total
    print(f"테스트 결과: loss={test_loss:.4f}, accuracy={test_acc:.4f}")

    wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})

    # ── 6. 최종 모델 저장 ──
    model_dir = Path(L_GM)
    model_dir.mkdir(parents=True, exist_ok=True)
    stem = f"gloss_mamba_{date_idx}"

    # 6-1. 전체 모델 (.pth)
    final_path = model_dir / f"{stem}.pth"
    torch.save(model.state_dict(), final_path)
    print(f"최종 모델 저장: {final_path}")

    # 6-2. 경량화 모델
    #   1순위: Dynamic Quantization int8 (Linux/FBGEMM, macOS/QNNPACK)
    #   2순위: float16 half-precision (int8 엔진 미지원 환경 폴백, ~50% 크기 감소)
    print("\n경량화 모델 생성 중...")
    cpu_model = model.cpu().eval()
    quantized_path = model_dir / f"{stem}.quantized.pt"

    try:
        # macOS(ARM/Intel) 는 qnnpack, Linux/Windows 는 fbgemm
        engine = "qnnpack" if torch.backends.quantized.engine == "none" else torch.backends.quantized.engine
        torch.backends.quantized.engine = engine
        with torch.no_grad():
            quantized_model = torch.ao.quantization.quantize_dynamic(
                cpu_model,
                qconfig_spec={nn.Linear},
                dtype=torch.qint8,
            )
        torch.save(quantized_model, quantized_path)
        method = "Dynamic Quantization (int8)"
    except Exception as e:
        print(f"  int8 quantization 실패 ({e}), float16 half-precision으로 폴백합니다.")
        half_state_dict = {k: v.half() for k, v in cpu_model.state_dict().items()}
        torch.save(half_state_dict, quantized_path)
        method = "float16 half-precision"

    orig_mb      = final_path.stat().st_size / 1024 / 1024
    quantized_mb = quantized_path.stat().st_size / 1024 / 1024
    print(f"경량화 모델 저장 [{method}]: {quantized_path}")
    print(f"  원본: {orig_mb:.2f} MB  →  경량화: {quantized_mb:.2f} MB  ({100*(1-quantized_mb/orig_mb):.0f}% 감소)")

    # 6-3. 업로드
    for path in (final_path, quantized_path):
        upload_file(
            local_root_path=str(model_dir),
            upload_path=str(model_dir),
            file_name=path.name,
        )

    artifact = wandb.Artifact(
        name="gloss-mamba-model",
        type="model",
        description="Trained CNN+Mamba gloss model (full + quantized)",
        metadata=dict(wandb.config),
    )
    artifact.add_file(str(final_path))
    artifact.add_file(str(quantized_path))
    wandb.log_artifact(artifact)

    wandb.finish()
    return model


# ──────────────────────────────────────────────
# Argument Parser
# ──────────────────────────────────────────────

def get_model_args():
    """
    실행 예시:
        python -m train.mamba_train --lr 0.0001 --bs 32 --epochs 300 --d_model 32
    """
    parser = argparse.ArgumentParser(description="CNN+Mamba2 수어 인식 학습")
    parser.add_argument("--lr", "--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--bs", "--batch_size",    type=int,   default=BATCH_SIZE)
    parser.add_argument("-e", "--epochs",          type=int,   default=EPOCHS)
    parser.add_argument("--wd", "--weight_decay",  type=float, default=WEIGHT_DECAY)
    parser.add_argument("--msl", "--max_sequence_len", type=int, default=CROP_LEN)
    parser.add_argument("--d_model",  type=int, default=UMAP_OUTPUT_DIM, help="모델 내부 차원 (기본: UMAP_OUTPUT_DIM=32)")
    parser.add_argument("--n_stages", type=int, default=2,  help="(Conv×3 + Mamba) 스테이지 수")
    parser.add_argument("--d_state",  type=int, default=16, help="SSM hidden state 차원")

    args, _ = parser.parse_known_args()
    print("하이퍼파라미터:")
    for k, v in vars(args).items():
        flag = "[default]" if v == parser.get_default(k) else ""
        print(f"  {flag} {k}: {v}")
    return args


if __name__ == "__main__":
    gpus = torch.cuda.device_count()
    if gpus:
        print(f"GPU 사용 중 ({gpus}개): {[torch.cuda.get_device_name(i) for i in range(gpus)]}")
    elif torch.backends.mps.is_available():
        print("Apple MPS 사용 중")
    else:
        print("WARNING: GPU 없음 — CPU로 학습합니다.")

    args = get_model_args()
    train_model(
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.bs,
        weight_decay=args.wd,
        max_sequence_len=args.msl,
        d_model=args.d_model,
        n_stages=args.n_stages,
        d_state=args.d_state,
    )
