"""
ASL 사전학습 가중치를 KSL 파인튜닝 모델로 전이합니다.

전이 전략:
  - 레이어 이름 + 가중치 shape가 모두 일치하는 경우에만 복사
  - stem_conv: ASL (708/1116, 192) vs KSL (540, 192) → shape 불일치 → 스킵 (랜덤 초기화)
  - classifier: ASL (384, 250) vs KSL (384, 33) → shape 불일치 → 스킵 (랜덤 초기화)
  - 그 외 모든 backbone 레이어 → 전이

실행:
  cd sign-language-korean
  python -m train.transfer_weights
"""
import os
import sys
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import keras

# 프로젝트 루트를 sys.path에 추가
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from src.backbones.transformer_backbone import (
    ECA, LateDropout, CausalDWConv1D, MultiHeadSelfAttention,
)
from src.backbones.asl_finetune_backbone import get_asl_finetuned_model
from src.asl_finetune_config import (
    ASL_WEIGHTS_PATH, CHANNELS_ASL, KSL_FINETUNE_DIM,
    MAX_LEN, FINETUNE_CKPT_DIR,
)
from src.config import NUM_CLASSES

# 이전 버전 Keras 호환 (quantization_config 제거)
_orig_dense_from_config = keras.layers.Dense.from_config.__func__
@classmethod
def _compat_dense_from_config(cls, config):
    config = {k: v for k, v in config.items() if k != 'quantization_config'}
    return _orig_dense_from_config(cls, config)
keras.layers.Dense.from_config = _compat_dense_from_config


def load_asl_model(weights_path: str) -> keras.Model:
    """ASL .h5 모델 로드 (custom objects 포함)."""
    custom_objects = {
        'ECA': ECA,
        'LateDropout': LateDropout,
        'CausalDWConv1D': CausalDWConv1D,
        'MultiHeadSelfAttention': MultiHeadSelfAttention,
    }
    print(f"[*] ASL 모델 로딩: {weights_path}")
    try:
        model = keras.models.load_model(weights_path, custom_objects=custom_objects, compile=False)
    except Exception as e:
        print(f"[!] 로드 실패: {e}")
        raise
    print(f"[*] ASL 모델 로드 완료. 레이어 수: {len(model.layers)}")
    return model


def transfer_weights(asl_model: keras.Model, ksl_model: keras.Model) -> dict:
    """
    이름 + shape 매칭으로 ASL → KSL 가중치 전이.

    Returns:
        {'transferred': [...], 'skipped': [...], 'not_found': [...]}
    """
    asl_by_name = {layer.name: layer for layer in asl_model.layers if layer.get_weights()}

    result = {'transferred': [], 'skipped': [], 'not_found': []}

    print("\n[*] 가중치 전이 시작...\n")
    for ksl_layer in ksl_model.layers:
        if not ksl_layer.get_weights():
            continue

        if ksl_layer.name not in asl_by_name:
            result['not_found'].append(ksl_layer.name)
            continue

        asl_layer = asl_by_name[ksl_layer.name]
        asl_w = asl_layer.get_weights()
        ksl_w = ksl_layer.get_weights()

        # 가중치 개수 확인
        if len(asl_w) != len(ksl_w):
            result['skipped'].append(
                f"{ksl_layer.name} (weight count: ASL={len(asl_w)}, KSL={len(ksl_w)})"
            )
            continue

        # Shape 일치 확인
        if not all(a.shape == k.shape for a, k in zip(asl_w, ksl_w)):
            shapes = [(a.shape, k.shape) for a, k in zip(asl_w, ksl_w)]
            result['skipped'].append(f"{ksl_layer.name} (shape mismatch: {shapes})")
            continue

        ksl_layer.set_weights(asl_w)
        result['transferred'].append(ksl_layer.name)

    # 결과 출력
    print("=" * 60)
    print(f"✅ 전이 성공: {len(result['transferred'])}개")
    for name in result['transferred']:
        print(f"   OK  {name}")

    print(f"\n⏭️  스킵 (shape 불일치): {len(result['skipped'])}개")
    for info in result['skipped']:
        print(f"   SKIP  {info}")

    print(f"\n❓ 이름 미매칭: {len(result['not_found'])}개")
    for name in result['not_found']:
        print(f"   MISS  {name}")
    print("=" * 60)

    return result


def main():
    keras.mixed_precision.set_global_policy("mixed_float16")

    # 1. KSL 파인튜닝 모델 빌드 (먼저 빌드해야 레이어 이름이 초기 번호로 할당됨)
    print("[*] KSL 파인튜닝 모델 빌드 중...")
    ksl_model = get_asl_finetuned_model(
        max_len=MAX_LEN,
        num_classes=NUM_CLASSES,
        input_channels=CHANNELS_ASL,
        latent_dim=KSL_FINETUNE_DIM,
    )
    print(f"[*] KSL 모델 입력 shape: {ksl_model.input_shape}")
    print(f"[*] KSL 모델 출력 shape: {ksl_model.output_shape}")
    ksl_model.summary(line_length=100)

    # 2. ASL 모델 로드
    if not Path(ASL_WEIGHTS_PATH).exists():
        print(f"[!] ASL 가중치 파일 없음: {ASL_WEIGHTS_PATH}")
        print("    sign-language/models/ 폴더를 확인하세요.")
        sys.exit(1)

    asl_model = load_asl_model(ASL_WEIGHTS_PATH)

    # 3. 가중치 전이
    result = transfer_weights(asl_model, ksl_model)

    if len(result['transferred']) == 0:
        print("\n[!] 경고: 전이된 레이어가 없습니다. 레이어 이름을 확인하세요.")
        print("[!] ASL 모델 레이어 이름:")
        for l in asl_model.layers:
            if l.get_weights():
                print(f"    {l.name}")

    # 4. 전이 완료 모델 저장
    save_path = str(Path(FINETUNE_CKPT_DIR) / "ksl_asl_transfer_init.keras")
    print(f"\n[*] 저장 중: {save_path}")
    ksl_model.save(save_path)
    print(f"[*] 완료. 다음 단계: python -m train.asl_finetune_train")

    return save_path


if __name__ == "__main__":
    main()
