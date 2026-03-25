#!/bin/bash

chmod +x ./tools/auto_pilot.sh
NOW=$(date +"%m%d_%H%M")

# 10번 반복 실행
for i in {1..10}
do
    echo "=========================================================="
    echo "  [Iteration $i / 10] 최적 하이퍼파라미터 탐색 시작"
    echo "=========================================================="

    # 1. 탐색 스크립트 실행 및 실시간 로그 노출
    # 로그에서 RESULT_ARGS: 로 시작하는 줄을 찾아 파라미터만 추출 - grep으로 줄을 찾고, cut으로 태그 뒤의 내용만 가져옵니다.
    BEST_ARGS=$(python -u -m train.transformer_best_params_search \
                    -m transformer \
                    --name "gloss_transformer_0325_${NOW}_${i}" \
                    --sr 0.4 \
                    --dr n \
                    --nt 10 2>&1 | tee /dev/stderr | grep "RESULT_ARGS:" | tail -n 1 | cut -d':' -f2-)

    # 추출 확인
    if [ -z "$BEST_ARGS" ]; then
        echo " [!] 에러: 최적 파라미터를 찾지 못했습니다. 로그를 확인하세요."
        exit 1
    fi

    echo " [?] 추출된 최적 파라미터: $BEST_ARGS"

    # 2. 추출된 파라미터로 실제 학습 스크립트 실행
    echo " [+] 실제 모델 학습 시작..."
    python -u -m train.gloss_transformer_train $BEST_ARGS

    echo " [OK] $i 번째 사이클 완료!"
    echo "=========================================================="
    echo ""
done