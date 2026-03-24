import importlib
importlib.import_module("src.config")
import os
from pathlib import Path
import tensorflow as tf
import numpy as np

from load_data.create_dataset import DataSetter
# custom layers
from src.backbone import get_model
from src.config import UMAP_OUTPUT_DIM, NUM_CLASSES, UMAP_LOAD_PATH, L_PARAMS, L_GM


def compare_tflite_with_keras(
        model_path: Path, tflite_path: Path,
        max_seq_len: int, batch_size: int
):
    # 1. dataset 로드
    _, _, test_ds = DataSetter(
        umap_path=UMAP_LOAD_PATH,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        dim_reduction=True
    ).get_datasets()

    # 2. Keras 모델 로드 및 가중치 설정
    model = get_model(max_len=max_seq_len, dropout_step=0, dim=UMAP_OUTPUT_DIM, num_classes=NUM_CLASSES, training=False)
    try:
        model.load_weights(model_path)
        print(f"[+] Keras Model loaded successfully.")
    except Exception as e:
        print(f"[-] Keras load failed: {e}")
        return

    # 3. TFLite 인터프리터 설정
    try:
        # from tensorflow.lite.python.interpreter import OpResolverType
        interpreter = tf.lite.Interpreter(
            model_path=str(tflite_path),
            # experimental_delegates=[tf.lite.experimental.load_delegate('flex_delegate.dll')] if os.name == 'nt' else None,
            # experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        )
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"[+] TFLite Model loaded successfully.")
    except Exception as e:
        print(f"[-] TFLite load failed: {e}")
        return

    # 4. 비교 테스트 시작
    print("\n[*] Starting Comparison (Keras vs TFLite)...")
    # 테스트 세트에서 배치 하나 가져오기
    for inputs, labels in test_ds.take(1):
        # Keras 예측
        keras_preds = model.predict(inputs, verbose=0)

        # TFLite 예측 (배치 내 첫 번째 샘플만 테스트)
        # TFLite는 기본적으로 1개씩 처리하므로 루프를 돌립니다.
        tflite_preds = []
        for i in range(len(inputs)):
            single_input = np.expand_dims(inputs[i], axis=0).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], single_input)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            tflite_preds.append(output_data[0])

        tflite_preds = np.array(tflite_preds)

        # 5. 수치 비교
        mse = np.mean(np.power(keras_preds - tflite_preds, 2))
        max_diff = np.max(np.abs(keras_preds - tflite_preds))

        print(f"\n[Comparison Results]")
        print(f"  > Mean Squared Error: {mse:.8f}")
        print(f"  > Max Absolute Difference: {max_diff:.8f}")

        # 첫 번째 샘플의 클래스 확률 비교 (Top 3)
        print(f"\n[Sample 0 Probability Comparison]")
        print(f"  Keras Top 3: {np.argsort(keras_preds[0])[-3:][::-1]}")
        print(f"  TFLite Top 3: {np.argsort(tflite_preds[0])[-3:][::-1]}")

        if max_diff < 1e-4:
            print("\n✅ Verification Successful: Both models produce nearly identical results.")
        else:
            print("\n⚠️ Warning: Significant difference detected. Check quantization or custom layers.")
        break

def evaluating_model_with_weights(
        learning_rate: float,
        model_path: Path,
        max_seq_len: int, batch_size: int
):
    # 1. dataset 로드
    _, _, test_ds = DataSetter(
        umap_path=UMAP_LOAD_PATH,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        dim_reduction=True
    ).get_datasets()

    # 데이터셋 샘플 검사
    print("[*] 데이터셋 레이블 범위 검사 중...")
    for _, labels in test_ds.take(5):
        lbl_min = np.min(labels)
        lbl_max = np.max(labels)
        if lbl_min < 0 or lbl_max >= 33:
            print(f"  > 레이블 범위 오류 발견! (Min: {lbl_min}, Max: {lbl_max})")
            raise ValueError("레이블이 0 ~ 32 범위를 벗어났습니다.")

    # 2. Keras 모델 로드 및 가중치 설정
    model = get_model(max_len=max_seq_len, dropout_step=0, dim=UMAP_OUTPUT_DIM, num_classes=NUM_CLASSES, training=False)
    try:
        model.load_weights(model_path)
        print(f"[+] Keras Model loaded successfully.")
    except Exception as e:
        print(f"[-] Keras load failed: {e}")
        return

    for i, weights in enumerate(model.get_weights()):
        if np.isnan(weights).any() or np.isinf(weights).any():
            print(f"🚨 경고: layer {i}번째 가중치에 NaN/inf가 포함되어 있습니다!")

    import keras
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    print(f"[*] Model compiled successfully!")
    print(f"    Input shape: {model.input_shape}")
    print(f"    Output shape: {model.output_shape}")

    print(f"[*] Loading model weights from: {model_path}")

    print("[*] Starting Evaluation...")
    eval_results = model.evaluate(test_ds, return_dict=True)
    print("Model Test Results: ")
    print(*(f"  > {k}: {v:.3f}" for k, v in eval_results.items()), sep='\n')

if __name__=="__main__":
    best_param_file_names = ["transformer-class=33-data=633-trial=10-2026_03_23_04-41.json"]
    best_param_files = [(a, Path(L_PARAMS) / a) for a in best_param_file_names]
    for n, f in best_param_files:
        import json

        with open(f, "r") as bpf:
            data = json.load(bpf)
        print(f"{n} 파일 오픈 완료!")
        print(*(f"    > {k}: {v}" for k, v in data.items()), sep='\n')
        # match = re.search(r'\d{4}_\d{2}_\d{2}_\d{2}-\d{2}',n)
        gm_path = Path(L_GM) / "gloss_transformer_2026_03_23_04-47.keras"
        if gm_path.exists():
            # suffix = match.group()
            # basename = f"gloss_transformer_{suffix}"
            # model_filename = f"{basename}.keras"
            # model_path = Path(L_CKPT) / model_filename
            # if os.path.exists(model_path):
            #     print(f"[*] 모델 {model_path} 존재!")
            if os.path.exists(gm_path):
                print(f"[*] 모델 {gm_path} 존재!")
                try:
                    evaluating_model_with_weights(
                        model_path=gm_path, # model_path,
                        learning_rate=data['learning_rate'],
                        max_seq_len=data["sequence_length"],
                        batch_size=data["batch_size"]
                    )
                except Exception as e:
                    print(f"[!] 모델 로드 중 오류 발생: {e}")
            else:
                print(f"[!] {gm_path} 파일을 찾을 수 없습니다")
        else:
            print(f"[!] {n} 파일에 날짜 suffix가 존재하지 않습니다.")

    # TFLite 비교
    # basename = f"gloss_transformer_2026_03_23_03-39"
    # model_filename = f"{basename}.keras"
    # model_tflite = f"{basename}.tflite"
    # model_path = Path(L_GM) / model_filename
    # tflite_path = Path(L_GM) / model_tflite # tflite 경로 설정
    # if os.path.exists(model_path):
    #     print(f"[*] 모델 {model_path} 존재!")
    #     try:
    #         evaluating_model_with_weights(
    #             model_path=model_path,
    #             tflite_path=tflite_path,
    #             max_seq_len=327,
    #             batch_size=32
    #         )
    #     except Exception as e:
    #         print(f"[!] 모델 로드 중 오류 발생: {e}")
    # else:
    #     print(f"[!] {model_filename} 파일을 찾을 수 없습니다")
