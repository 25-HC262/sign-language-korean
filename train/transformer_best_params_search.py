import argparse
import importlib

from optuna_integration import KerasPruningCallback

importlib.import_module("src.tf_keras_config")

import keras
keras.mixed_precision.set_global_policy("mixed_float16") # fp16 가속 keras3 버전, 모델 구성 & 학습 시에만 사용
import keras.optimizers
import optuna
from optuna.storages import RDBStorage

from src.config import OPTUNA_TRIALS_PATH, NUM_CLASSES, SUBSET_RATIO, \
    OPTUNA_MODEL, OPTUNA_STUDY_NAME, N_TRIALS, UMAP_OUTPUT_DIM, \
    L_TOOLS, VALIDATION_SPLIT, get_base_parser, L_PARAMS, \
    PathConfig, OUTPUT_DIM

from load_data.create_dataset import DataSetter, upload_file
from src.backbones.transformer_backbone import get_model


def objective(trial):
    # 1. 하이퍼파라미터 탐색 공간 정의
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 0.2)

    num_train_epochs = 200 # 고정
    sequence_length = 300  # 고정

    # 2. 데이터 불러오기
    train_dataset, val_dataset, _ = DataSetter(
        umap_path=umap_path,
        max_seq_len=sequence_length,
        batch_size=batch_size,
        dim_reduction=dim_rd
    ).get_datasets()
    # 2-1. 학습 데이터 줄이기
    num_train = int(80*NUM_CLASSES*(1-VALIDATION_SPLIT) * subset_ratio)
    train_dataset = train_dataset.shuffle(buffer_size=1000, seed=42)
    small_train_dataset = train_dataset.take(max(num_train,1))
    trial.set_user_attr("num_train", num_train) # trial에 저장

    # 2-2. 검증 데이터 줄이기
    num_val = int(80*NUM_CLASSES*VALIDATION_SPLIT * subset_ratio)
    val_dataset = val_dataset.shuffle(buffer_size=1000, seed=42)
    small_val_dataset = val_dataset.take(max(num_val,1))

    output = UMAP_OUTPUT_DIM if dim_rd else OUTPUT_DIM

    # 3. 모델 설정
    model = get_model(max_len=sequence_length, dropout_step=0, dim=output, num_classes=NUM_CLASSES)
    model.compile(
        optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
        loss = keras.losses.SparseCategoricalCrossentropy(),
        metrics = ['accuracy']
    )
    print("Model compile finished.")

    # 4. 모델 학습
    model.fit(
        small_train_dataset,
        validation_data=small_val_dataset,
        epochs=num_train_epochs,
        verbose=1,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                pc.LOCAL_PATHS["optuna_gm_ckpt"],
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
            ),
            KerasPruningCallback(trial, 'val_loss') # 에폭 종료 시마다 'val_loss' optuna에 보고
        ]
    )

    # Pruning이 발생하면 'val_loss'가 NaN이 될 수 있으므로 체크
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    # 5. 모델 평가
    eval_result = model.evaluate(small_val_dataset, return_dict=True)

    # 정확도(accuracy) 반환
    return eval_result['accuracy']

if __name__=="__main__":
    def get_optuna_config():
        base_parser = get_base_parser()
        parser = argparse.ArgumentParser(parents=[base_parser])
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
        # 유맵 사용 여부
        parser.add_argument(
            "--dr", "--dim_reduction",
            choices=['n', 'y'],
            required=True
        )

        o_args = parser.parse_args()
        print(*(f"   > {'[default]' if v==parser.get_default(k) else ''} {k}: {v} selected." for k,v in vars(o_args).items()), sep='\n')
        return o_args
    args = get_optuna_config()
    # 사용자 옵션
    study_name = args.name
    subset_ratio = args.sr
    n_trials = args.nt
    model_n = args.model
    dim_rd = True if args.dr=='y' else False

    pc = PathConfig(args)

    umap_path = pc.UMAP_LOAD_PATH if dim_rd else None

    # 2. optuna 세팅
    storage = RDBStorage(url=OPTUNA_TRIALS_PATH)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner() # 중간값보다 성적 나쁘면 조기 종료
    )

    print(f" ============== parameter trials {n_trials}번 시도 시작! ============== ")
    study.optimize(objective, n_trials=n_trials)

    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")

    # 3. 최적 파라미터 추출
    import json, datetime
    date_idx = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M")
    best_num_train = study.best_trial.user_attrs.get("num_train", "N/A") # 사용 데이터 개수
    BEST_PARAMS_PATH = L_PARAMS / f"{pc.SELECTED_GM_TYPE}-class={NUM_CLASSES}-data={best_num_train}-trial={n_trials}-{date_idx}.json"
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(study.best_params, f)

    upload_file(local_root_path=str(BEST_PARAMS_PATH.parent), upload_path=pc.LOAD_TOOLS, file_name=str(BEST_PARAMS_PATH.name))

    # 4. 성적 순 상위 30%의 정확도 & 소요 시간 출력
    df = study.trials_dataframe()
    top_num = max(int(n_trials*0.3), 1)
    top_trials = df.sort_values(by="value", ascending=False).head(top_num)
    print(f"--------- Top {top_num} Trials --------- ")
    print(top_trials[['number', 'value', 'duration', 'params_learning_rate', 'params_batch_size', 'params_num_train_epochs', 'params_weight_decay', 'params_sequence_length']])

    import matplotlib.pyplot as plt
    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_intermediate_values,
        plot_param_importances
    )
    plt.style.use('ggplot')

    # 4. 분석 결과 도출
    # 4-1. 맵 설정 (함수 객체 자체를 저장하여 루프에서 실행)
    # (함수, 제목, 저장파일명)
    base_filename = f"{pc.SELECTED_GM_TYPE}-{study_name}-{date_idx}-"
    main_info = f"Study: {study_name} | Best: {study.best_value:.4f} | Trials: {n_trials}"
    sub_info = f"N_Train: {best_num_train} | Classes: {NUM_CLASSES}"
    opt_map = {
        'H': (plot_optimization_history, "Optimization History", base_filename+"optimization_history.png"),
        'I': (plot_intermediate_values, "Intermediate Values", base_filename+"intermediate_values.png"),
        'P': (plot_param_importances, "Hyperparameter Importances", base_filename+"param_importances.png")
    }
    for key in ['H', 'I', 'P']:
        plot_func, graph_type_name, filename = opt_map.get(key, opt_map['H'])
        try:
            print(f"Drawing {graph_type_name}...")

            # 4-2. 함수 실행
            ax = plot_func(study)
            fig = plt.gcf()

            # 플롯 제목
            full_title = f"{main_info}\n{sub_info}\n[{graph_type_name}]"
            plt.title(full_title, fontsize=9, pad=20)

            # 우측에 상세 정보 리스트 추가
            info_text = (
                f"[Experiment Info]\n"
                f"Date: {date_idx}\n"
                f"Trials: {n_trials}\n"
                f"Best Params:\n"
                f"- Learning Rate: {study.best_params.get('learning_rate', 0):.2e}\n"
                f"- Batch Size: {study.best_params.get('batch_size', 0)}\n"
                f"- Weight Decay: {study.best_params.get('weight_decay', 0):.2f}\n"
                f"- Seq Len: {study.best_params.get('sequence_length', 0)}"
            )
            # 그래프 오른쪽에 텍스트 박스 배치
            fig.text(1.02, 0.5, info_text, transform=ax.transAxes,
                     verticalalignment='center', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

            # 레이아웃 조정 (오른쪽 텍스트가 잘리지 않게 여백 확보)
            plt.tight_layout(rect=[0, 0, 0.85, 0.95])

            # 4-3. 경로 결합 및 저장
            save_path = L_TOOLS / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

            print(f"    > {graph_type_name} 저장 완료: {save_path}")

            # 메모리 해제 및 캔버스 초기화
            plt.clf()
            plt.close('all')

            # 업로드
            upload_file(local_root_path=str(save_path.parent), upload_path=pc.LOAD_TOOLS, file_name=str(save_path.name))

        except Exception as e:
            print(f"    > {graph_type_name} 플롯 생성 실패: {e}")

        print("\n그래프 저장 완료!")

    # --- 추가: Bash 스크립트가 읽기 편하도록 최적 파라미터를 한 줄로 출력 ---
    bp = study.best_params
    # 학습 스크립트의 인자명(--lr, --bs 등)에 맞춰서 문자열 생성
    cmd_args = f"--lr {bp['learning_rate']} --bs {bp['batch_size']} --epochs {bp['num_train_epochs']} --wd {bp['weight_decay']} --msl {bp['sequence_length']}"

    # 식별하기 쉽게 특수한 태그(RESULT_ARGS:)를 붙여서 출력
    print(f"\nRESULT_ARGS: {cmd_args}")
