#!/bin/bash

# 100번 반복 실행
for i in {1..100}
do
    echo "=========================================================="
    echo "  [Iteration $i / 100] 최적 하이퍼파라미터 탐색 시작"
    echo "=========================================================="

    # 1. 탐색 스크립트 실행 및 로그 저장
    # --nt 10 정도로 짧게 서치하거나 필요에 따라 조절하세요.
    SEARCH_LOG=$(python -m train.transformer_best_params_search \
        -m transformer \
        --name "gloss_transformer_0325_${i}" \
        --sr 0.4 \
        --nt 10)
    # 2. 로그에서 RESULT_ARGS: 로 시작하는 줄을 찾아 파라미터만 추출
    # grep으로 줄을 찾고, cut으로 태그 뒤의 내용만 가져옵니다.
    BEST_ARGS=$(echo "$SEARCH_LOG" | grep "RESULT_ARGS:" | cut -d':' -f2-)

    if [ -z "$BEST_ARGS" ]; then
        echo " [!] 에러: 최적 파라미터를 찾지 못했습니다. 로그를 확인하세요."
        exit 1
    fi

    echo " [?] 추출된 최적 파라미터: $BEST_ARGS"

    # 3. 추출된 파라미터로 실제 학습 스크립트 실행
    echo " [+] 실제 모델 학습 시작..."
    python -m train.gloss_transformer_train $BEST_ARGS

    echo " [OK] $i 번째 사이클 완료!"
    echo "=========================================================="
    echo ""
done