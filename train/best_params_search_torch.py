"""
best_params_search.py — CNN + Mamba2 수어 인식 모델 하이퍼파라미터 탐색 스크립트 (PyTorch + Optuna)

기존 Keras 버전을 PyTorch Mamba 모델에 맞게 재작성.
TF Dataset → numpy → PyTorch DataLoader 브릿지 사용 (mamba_train.py와 동일).

실행 예시:
    python -m train.best_params_search -m mamba --name mamba_optuna_0326 --sr 0.3 --nt 10 --dr y
"""

import argparse
import importlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import optuna
from optuna.storages import RDBStorage

from src.config import (
    OPTUNA_TRIALS_PATH, EPOCHS, NUM_CLASSES, SUBSET_RATIO,
    OPTUNA_MODEL, OPTUNA_STUDY_NAME, N_TRIALS, UMAP_OUTPUT_DIM,
    L_TOOLS, VALIDATION_SPLIT, L_PARAMS,
    PathConfig, CROP_LEN, OUTPUT_DIM, get_config_args,
)
from load_data.create_dataset import DataSetter, upload_file


# ──────────────────────────────────────────────
# TF Dataset → PyTorch Dataset 브릿지 (mamba_train.py에서 재사용)
# ──────────────────────────────────────────────

class TFDatasetBridge(Dataset):
    def __init__(self, tf_dataset, max_seq_len: int = CROP_LEN):
        print("  TF Dataset → numpy 변환 중...")
        sequences, labels = [], []
        for x, y in tf_dataset.unbatch().as_numpy_iterator():
            sequences.append(x[:max_seq_len])
            labels.append(int(y))
        self.sequences = np.array(sequences, dtype=np.float32)
        self.labels    = np.array(labels,    dtype=np.int64)
        print(f"  변환 완료: {len(self.labels)}개 샘플, shape={self.sequences.shape}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx]), torch.tensor(self.labels[idx])


# ──────────────────────────────────────────────
# Optuna Objective
# ──────────────────────────────────────────────

def objective(trial):
    # 1. 하이퍼파라미터 탐색 공간 정의
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size    = trial.suggest_categorical("batch_size", [16, 32, 64])
    weight_decay  = trial.suggest_float("weight_decay", 0.0, 0.3)
    d_model       = trial.suggest_categorical("d_model", [32, 64, 128])
    n_stages      = trial.suggest_int("n_stages", 1, 3)
    d_state       = trial.suggest_categorical("d_state", [16, 32, 64])

    sequence_length = CROP_LEN

    # 2. 데이터 불러오기
    train_tf, val_tf, _ = DataSetter(
        umap_path=umap_path,
        max_seq_len=sequence_length,
        batch_size=batch_size,
        dim_reduction=dim_rd,
    ).get_datasets()

    # 2-1. 학습 데이터 줄이기
    num_train = int(80 * NUM_CLASSES * (1 - VALIDATION_SPLIT) * subset_ratio)
    train_tf = train_tf.shuffle(buffer_size=1000, seed=42).take(max(num_train, 1))
    trial.set_user_attr("num_train", num_train)

    # 2-2. 검증 데이터 줄이기
    num_val = int(80 * NUM_CLASSES * VALIDATION_SPLIT * subset_ratio)
    val_tf = val_tf.shuffle(buffer_size=1000, seed=42).take(max(num_val, 1))

    # TF → PyTorch 변환
    train_ds = TFDatasetBridge(train_tf, sequence_length)
    val_ds   = TFDatasetBridge(val_tf,   sequence_length)

    val_available = len(val_ds) > 0
    if not val_available:
        print("Warning: val_dataset이 비어있습니다. validation 없이 학습을 진행합니다.")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    output_dim = UMAP_OUTPUT_DIM if dim_rd else OUTPUT_DIM

    # 3. 모델 설정
    model_module = importlib.import_module(f"src.backbones.{model_n}_backbone")
    get_model = getattr(model_module, "get_model")
    model = get_model(
        max_len=sequence_length,
        in_dim=output_dim,
        d_model=d_model,
        n_stages=n_stages,
        d_state=d_state,
        num_classes=NUM_CLASSES,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    dynamic_min_lr = learning_rate * 0.01
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=dynamic_min_lr
    )
    print(f"모델 설정 완료. device={device}")

    # 4. 학습 루프
    best_val_loss = float("inf")
    best_val_acc  = 0.0
    patience_counter = 0
    early_stop_patience = 10

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        if not val_available:
            continue

        # --- Validate ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss    += criterion(logits, y).item() * len(y)
                val_correct += (logits.argmax(dim=1) == y).sum().item()
                val_total   += len(y)

        val_loss /= val_total
        val_acc   = val_correct / val_total

        scheduler.step(val_loss)
        print(f"  Epoch {epoch:3d}/{EPOCHS} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # Optuna 중간값 보고 + Pruning
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Early stopping & best 갱신
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"  Early stopping (patience={early_stop_patience})")
                break

    return best_val_acc


if __name__ == "__main__":
    def get_optuna_config():
        """
        명령어 예시: python -m train.best_params_search -m mamba --name mamba_optuna_0326 --sr 0.3 --nt 10 --dr y
        """
        base_parser, _ = get_config_args()
        parser = argparse.ArgumentParser(parents=[base_parser])
        parser.add_argument("-m", "--model",           default=OPTUNA_MODEL)
        parser.add_argument("-n", "--name", "--study_name", default=OPTUNA_STUDY_NAME)
        parser.add_argument("--sr", "--subset_ratio",  type=float, default=SUBSET_RATIO)
        parser.add_argument("--nt", "--n_trials",      type=int,   default=N_TRIALS)
        parser.add_argument("--dr", "--dim_reduction", choices=['n', 'y'])

        final_args = parser.parse_args()
        print(*(f"   > {'[default]' if v == parser.get_default(k) else ''} {k}: {v} selected."
                for k, v in vars(final_args).items()), sep='\n')
        return final_args

    args         = get_optuna_config()
    study_name   = args.name
    subset_ratio = args.sr
    n_trials     = args.nt
    model_n      = args.model
    dim_rd       = args.dr == 'y'

    pc        = PathConfig(args)
    umap_path = pc.UMAP_LOAD_PATH if dim_rd else None

    # Optuna 세팅
    storage = RDBStorage(url=OPTUNA_TRIALS_PATH)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(),
    )

    print(f" ============== parameter trials {n_trials}번 시도 시작! ============== ")
    study.optimize(objective, n_trials=n_trials)

    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")

    # 최적 파라미터 저장
    import json, datetime
    date_idx = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M")
    best_num_train   = study.best_trial.user_attrs.get("num_train", "N/A")
    BEST_PARAMS_PATH = L_PARAMS / f"{pc.SELECTED_GM_TYPE}-class={NUM_CLASSES}-data={best_num_train}-trial={n_trials}-{date_idx}.json"
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(study.best_params, f, indent=2)

    upload_file(
        local_root_path=str(BEST_PARAMS_PATH.parent),
        upload_path=pc.LOAD_TOOLS,
        file_name=str(BEST_PARAMS_PATH.name),
    )

    # 상위 30% 결과 출력
    df      = study.trials_dataframe()
    top_num = max(int(n_trials * 0.3), 1)
    top_trials = df.sort_values(by="value", ascending=False).head(top_num)
    print(f"--------- Top {top_num} Trials --------- ")
    cols = ['number', 'value', 'duration',
            'params_learning_rate', 'params_batch_size', 'params_weight_decay',
            'params_d_model', 'params_n_stages', 'params_d_state']
    print(top_trials[[c for c in cols if c in top_trials.columns]])

    # 시각화
    import matplotlib.pyplot as plt
    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_intermediate_values,
        plot_param_importances,
    )
    plt.style.use('ggplot')

    base_filename = f"{pc.SELECTED_GM_TYPE}-{study_name}-{date_idx}-"
    main_info     = f"Study: {study_name} | Best: {study.best_value:.4f} | Trials: {n_trials}"
    sub_info      = f"N_Train: {best_num_train} | Classes: {NUM_CLASSES}"
    opt_map = {
        'H': (plot_optimization_history, "Optimization History",      base_filename + "optimization_history.png"),
        'I': (plot_intermediate_values,  "Intermediate Values",       base_filename + "intermediate_values.png"),
        'P': (plot_param_importances,    "Hyperparameter Importances", base_filename + "param_importances.png"),
    }

    bp = study.best_params
    for key in ['H', 'I', 'P']:
        plot_func, graph_type_name, filename = opt_map[key]
        try:
            print(f"Drawing {graph_type_name}...")
            ax  = plot_func(study)
            fig = plt.gcf()

            plt.title(f"{main_info}\n{sub_info}\n[{graph_type_name}]", fontsize=9, pad=20)

            info_text = (
                f"[Experiment Info]\n"
                f"Date: {date_idx}\n"
                f"Trials: {n_trials}\n"
                f"Best Params:\n"
                f"- LR: {bp.get('learning_rate', 0):.2e}\n"
                f"- Batch: {bp.get('batch_size', 0)}\n"
                f"- WD: {bp.get('weight_decay', 0):.2f}\n"
                f"- d_model: {bp.get('d_model', 0)}\n"
                f"- n_stages: {bp.get('n_stages', 0)}\n"
                f"- d_state: {bp.get('d_state', 0)}"
            )
            fig.text(1.02, 0.5, info_text, transform=ax.transAxes,
                     verticalalignment='center', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            plt.tight_layout(rect=[0, 0, 0.85, 0.95])

            save_path = L_TOOLS / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"    > {graph_type_name} 저장 완료: {save_path}")

            plt.clf()
            plt.close('all')

            upload_file(
                local_root_path=str(save_path.parent),
                upload_path=pc.LOAD_TOOLS,
                file_name=str(save_path.name),
            )
        except Exception as e:
            print(f"    > {graph_type_name} 플롯 생성 실패: {e}")

    print("\n그래프 저장 완료!")

    # Bash 스크립트용 최적 파라미터 한 줄 출력
    cmd_args = (
        f"--lr {bp['learning_rate']} --bs {bp['batch_size']} --wd {bp['weight_decay']} "
        f"--d_model {bp['d_model']} --n_stages {bp['n_stages']} --d_state {bp['d_state']}"
    )
    print(f"\nRESULT_ARGS: {cmd_args}")
