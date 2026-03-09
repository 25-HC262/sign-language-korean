import os

import matplotlib.pyplot as plt
import optuna
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_intermediate_values,
    plot_param_importances
)

# 설정 불러오기
from src.config import OPTUNA_TRIALS_PATH, L_TOOLS

# 1. Study 로드
study = optuna.load_study(study_name="gloss_transformer_0226", storage=OPTUNA_TRIALS_PATH)
# DB에 있는 모든 스터디 목록 가져오기
try:
    summaries = optuna.get_all_study_summaries(storage=OPTUNA_TRIALS_PATH)
    print("--- 현재 DB에 존재하는 스터디 목록 ---")
    for s in summaries:
        print(f"이름: {s.study_name} | Trial 횟수: {s.n_trials}")
    print("-----------------------------------")
except Exception as e:
    print(f"DB 읽기 실패: {e}")

plt.style.use('ggplot')

# 2. 맵 설정 (함수 객체 자체를 저장하여 루프에서 실행)
# (함수, 제목, 저장파일명)
opt_map = {
    'H': (plot_optimization_history, "Optimization History", "optimization_history.png"),
    'I': (plot_intermediate_values, "Intermediate Values", "intermediate_values.png"),
    'P': (plot_param_importances, "Hyperparameter Importances", "param_importances.png")
}

for key in ['H', 'I', 'P']:
    plot_func, name, filename = opt_map.get(key, opt_map['H'])
    try:
        print(f"Drawing {name}...")

        # 함수 실행 (여기서 에러가 나야 try-except가 잡아줍니다)
        ax = plot_func(study)

        plt.title(name, fontsize=14)
        plt.tight_layout()

        # 경로 결합 및 저장
        save_path = os.path.join(L_TOOLS, filename)
        plt.savefig(save_path, dpi=300)

        print(f"{name} 저장 완료: {save_path}")
        plt.show()

    except Exception as e:
        # Trial이 1개뿐이거나 Pruning 데이터가 없으면 여기서 걸러짐
        print(f"{name} 플롯 생성 실패: {e}")