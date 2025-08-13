import os
import sys, concurrent.futures
from pathlib import Path
from google.cloud import storage # needs `pip install --upgrade google-cloud-storage`
from tqdm import tqdm
import google.auth
from google.auth.transport.requests import Request

FILE_SUFFIX = "_keypoints.json"     # 업로드 대상 파일 패턴

# blob: Binary Large OBject
def upload_one(client, bucket_name: str, local_path: str, gcs_path: str):
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)

def upload_dir(root_dir, bucket_name, prefix=""):
    # prefix: upload할 파일 앞에 붙임
    client = storage.Client.from_service_account_json(r"C:\Users\chaei\Desktop\FullMoon\SMHH\gradproj\data\gcs_key.json") # google_application_credentials 환경변수 사용
    root_dir = Path(root_dir).expanduser().resolve()

    files = list(root_dir.rglob(f"*{FILE_SUFFIX}"))
    for f in files:
        print(f)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = []
        for path in files:
            rel = path.relative_to(root_dir).as_posix()
            gcs_path = f"{prefix}/{rel}" if prefix else rel
            futures.append(pool.submit(upload_one, client,bucket_name,str(path),gcs_path))

        for f in tqdm(concurrent.futures.as_completed(futures),
                      total=len(futures), desc="Uploading json"):
            f.result()

if __name__=="__main__":
    # if len(sys.argv) < 3:
    #     print("Usage: python upload_to_gcs.py <LOCAL_DIR> <BUCKET> [DEST_PREFIX]")
    #     sys.exit(1)
    local_dir = "data\openpose_keypoints"
    bucket_name = "sign-language-korean"
    prefix = "openpose_keypoints"
    # print("CWD = ", os.getcwd())
    upload_dir(local_dir, bucket_name, prefix)