import importlib
importlib.import_module("src.config")
import os
from pathlib import Path

import keras

from load_data.create_dataset import DataSetter
# custom layers
from src.transformer import get_model
from src.config import L_GM, UMAP_OUTPUT_DIM, NUM_CLASSES, L_CKPT, UMAP_LOAD_PATH

def evaluating_model_with_weights(
        model_path: Path,
        learning_rate: float, max_seq_len: int, batch_size: int
):
    train_ds, val_ds, test_ds = DataSetter(
        umap_path=UMAP_LOAD_PATH,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        dim_reduction=True
    ).get_datasets()

    model = get_model(max_len=max_seq_len, dropout_step=0, dim=UMAP_OUTPUT_DIM, num_classes=NUM_CLASSES)

    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    print(f"[*] Model compiled successfully!")
    print(f"    Input shape: {model.input_shape}")
    print(f"    Output shape: {model.output_shape}")

    print(f"[*] Loading model weights from: {model_path}")

    try:
        model.load_weights(model_path)
        print("[+] Model loaded successfully.")
    except Exception as e:
        print(f"[-] Model load failed: {e}")
        exit()

    print("[*] Starting Evaluation...")
    eval_results = model.evaluate(test_ds, return_dict=True)
    print("Model Test Results: ")
    print(*(f"  > {k}: {v:.3f}" for k, v in eval_results.items()), sep='\n')


if __name__=="__main__":
    best_param_file_names = ["best_params-2026_02_26_12-15.json", "best_params-2026_02_26_15-56.json"]
    best_param_files = [(a, Path(L_GM) / a) for a in best_param_file_names]
    for n, f in best_param_files:
        import json, re
        with open(f, "r") as bpf:
            data = json.load(bpf)
        print(f"{n} 파일 오픈 완료!")
        print(*(f"    > {k}: {v}" for k, v in data.items()), sep='\n')
        match = re.search(r'\d{4}_\d{2}_\d{2}_\d{2}-\d{2}',n)
        if match:
            suffix = match.group()
            model_filename = f"gloss_transformer_{suffix}.keras"
            model_path = Path(L_CKPT) / model_filename
            if os.path.exists(model_path):
                print(f"[*] 모델 {model_path} 존재!")
                try:
                    evaluating_model_with_weights(
                        model_path=model_path,
                        learning_rate=data["learning_rate"],
                        max_seq_len=data["sequence_length"],
                        batch_size=data["batch_size"]
                    )
                except Exception as e:
                    print(f"[!] 모델 로드 중 오류 발생: {e}")
            else:
                print(f"[!] {model_filename} 파일을 찾을 수 없습니다")
        else:
            print(f"[!] {n} 파일에 날짜 suffix가 존재하지 않습니다.")
