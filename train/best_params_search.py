import evaluate
import keras
import keras.optimizers

keras.mixed_precision.set_global_policy("mixed_float16") # fp16 가속 keras3 버전
import optuna
import tensorflow as tf
from optuna.storages import RDBStorage

from load_data.create_dataset import TrainDataLoader
from src.backbone import get_model
from src.config import OPTUNA_TRIALS_PATH, LOAD_DATA, EPOCHS, NUM_CLASSES, SUBSET_RATIO, \
    BEST_PARAMS_PATH, OPTUNA_MODEL, OPTUNA_STUDY_NAME, N_TRIALS, LOCAL_PATHS, UMAP_OUTPUT_DIM


def objective(trial):
    # 1. 하이퍼파라미터 탐색 공간 정의
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    num_train_epochs = trial.suggest_int("num_train_epochs", 30, EPOCHS)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.3)
    sequence_length = trial.suggest_int("sequence_length", 100, 380)

    # 2. 데이터 불러오기
    train_dataset, val_dataset, _ = loader.create_transformer_dataset(batch_size=batch_size, max_len=sequence_length)
    # 2-1. 학습 데이터 줄이기
    num_train = int(loader.train_size * subset_ratio)
    train_dataset = train_dataset.shuffle(buffer_size=1000, seed=42)
    small_train_dataset = train_dataset.take(num_train)

    # 2-2. 검증 데이터 줄이기
    num_val = int(loader.val_size * subset_ratio)
    val_dataset = val_dataset.shuffle(buffer_size=1000, seed=42)
    small_val_dataset = val_dataset.take(num_val)

    # 3. 모델 설정
    model = get_model(max_len=sequence_length, dropout_step=0, dim=UMAP_OUTPUT_DIM, num_classes=NUM_CLASSES)
    model.compile(
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
        loss = keras.losses.SparseCategoricalCrossentropy(),
        metrics = ['accuracy']
    )
    print("Model compile finished.")

    # 4. 모델 학습
    history = model.fit(
        small_train_dataset,
        validation_data=small_val_dataset,
        epochs=num_train_epochs,
        verbose=1,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                LOCAL_PATHS["gm_ckpt"],
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
    )

    # 5. 모델 평가
    eval_result = model.evaluate(small_val_dataset, return_dict=True)

    # 정확도(accuracy) 반환
    return eval_result['accuracy']

if __name__=="__main__":
    # 1. 사용 가능한 GPU 리스트 출력
    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        print("WARNING: 지금 GPU가 아니라 CPU를 쓰고 있습니다!")
    else:
        # 현재 인식된 GPU 개수와 상세 명칭 출력
        print(f"GPU 사용 중. 현재 사용 가능한 GPU 개수: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f" - GPU [{i}]: {gpu}")

    """
    TO-DO
    transformer에서 다른 모델들로 확장
    """
    def get_optuna_config():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-m", "--model",
            default=OPTUNA_MODEL
        )
        parser.add_argument(
            "-n", "--name", "--study_name",
            default=OPTUNA_STUDY_NAME
        )
        # 데이터 사용 비율 설정
        parser.add_argument(
            "--sr", "--subset_ratio",
            type=float,
            default=SUBSET_RATIO
        )
        # 탐색 횟수
        parser.add_argument(
            "--nt", "--n_trials",
            type=int,
            default=N_TRIALS
        )
        args, _ = parser.parse_known_args()
        print(*(f"   > {'[default]' if v==parser.get_default(k) else ''} {k}: {v} selected." for k,v in vars(args).items()), sep='\n')
        return args
    args = get_optuna_config()
    # 사용자 옵션
    study_name = args.name
    subset_ratio = args.sr
    n_trials = args.nt

    # optuna 세팅
    storage = RDBStorage(url=OPTUNA_TRIALS_PATH)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True
    )

    # 데이터 로드
    loader = TrainDataLoader(data_path=LOAD_DATA, is_training_transformer=True)

    print(f" ============== parameter trials {n_trials}번 시도 시작! ============== ")
    study.optimize(objective, n_trials=n_trials)

    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")

    # 최적 파라미터 추출
    import json, datetime
    date_idx = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M")
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(study.best_params, f)
