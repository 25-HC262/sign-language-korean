import tensorflow as tf
from google.cloud import storage

from src.config import PathConfig, L_PREPROCESSED_DATA  # PathConfig 등이 담긴 모듈 가정
from src.config import Trainer, L_DATA, get_config_args


def test_data_loader_standalone():
    from urllib.parse import urlparse
    from pathlib import Path

    path = "gs://test-openpose-keypoint"
    parsed = urlparse(path)

    print(f"1. netloc (bucket): '{parsed.netloc}'")
    print(f"    netloc into str: '{str(parsed.netloc)}'")
    print(f"2. path (prefix)  : '{parsed.path}'")
    print(f"3. lstrip('/')    : '{parsed.path.lstrip('/')}'")


    print("\n" + "="*50)
    print("      TEST DATA LOADER STANDALONE CHECK")
    print("="*50)
    print(f"target upload path: {pc.LOAD_PREPROCESSED_DATA}")

    # 1. Loader 설정 (사용자 파라미터 기반)
    loader_args = {
        "trainer": Trainer.GM,
        "data_path": L_DATA,            # 실제로는 쓰이지 않음 (test 로드 시)
        "test_data_path": pc.LOAD_TEST,  # gs://test-openpose-keypoint
        "dim_reduction": True,         # 테스트용이므로 False
        "umap_path": pc.UMAP_LOAD_PATH
    }

    print(f"[*] Target Test Path: {loader_args['test_data_path']}")

    # 2. 로더 객체 생성
    from load_data.create_dataset import TrainDataLoader, upload_file
    loader = TrainDataLoader(**loader_args)

    # 3. 테스트 데이터셋 생성 실행
    print("\n[*] Starting create_test_dataset()...")
    test_ds = loader.create_test_dataset()

    data_dir = "un=encoder-nc=33-dt=gloss-dd=2d"
    test_path = Path(L_PREPROCESSED_DATA) / data_dir

    tf.data.Dataset.save(test_ds, str(test_path))

    print(f"[*] 생성된 데이터셋을 클라우드로 업로드합니다... (Path: {test_path.name})")
    upload_file(local_root_path=str(test_path), upload_path=pc.LOAD_PREPROCESSED_DATA, file_name=None)

    # 4. 결과 확인
    print("\n" + "-"*30)
    if loader.videos:
        print(f"[SUCCESS] Total videos loaded: {len(loader.videos)}")
        # 첫 번째 데이터 샘플 확인
        sample = loader.videos[0]
        print(f"Sample Class Label: {sample['class_label']}")
        print(f"Sample Sequence Shape: {sample['sequence'].shape}")
    else:
        print("[FAIL] No videos found in the test path. Check GCS prefixes.")
    print("-"*30 + "\n")

if __name__ == "__main__":
    args = get_config_args()
    pc = PathConfig(args)
    test_data_loader_standalone()