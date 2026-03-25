#!/bin/bash

#sudo apt install -y p7zip-full jq     # 필수 도구 설치 확인

source .env
echo "$AIHUB_API_KEY"                  # AIHub에서 신청해야 함
DATASET_ID=103                        # 수어영상 데이터셋 아이디 103
GCS_BUCKET_NAME="gs://openpose-keypoint"   # 수어영상 키포인트는 "openpose-keypoints"

FILE_KEYS=(39583 39585 39586 39587 39588 39589 39590 39591 39592 39593 39594 39595 39596 39597 39598 39599)  # AIHub 페이지에서 찾을 수 있음.

# 위 FILE_KEYS와 1:1로 대응되는 폴더 숫자 목록
FILE_NUMBERS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)                             # EX. 08_real_sen_keypoint 폴더처럼 맨앞에 붙는 PREFIX

# 제외할 데이터 목록 (정규식 패턴으로 생성)
# 제외 사유: 이미 보유 중인 33개 데이터
EXCLUDE_LIST="0019|0021|0033|0035|0067|0109|0111|0133|0145|0181|0183|0185|0187|0189|0191|0193|0195|0197|0354|0355|0356|0357|1045|1613|1771|1773|1775|1940|1945|1976|1978|1999|2000"

DOWNLOAD_PATH="/mnt/disks/data/004.수어영상/1.Training/라벨링데이터/REAL/SEN"
mkdir -p "$DOWNLOAD_PATH"
echo "0. 현재 pwd가 /mnt/disks/data임을 확인"

# --- aihubshell 준비 ---
if [ ! -f "aihubshell" ]; then
    echo "1. aihubshell을 다운로드합니다."
    curl -L -o "aihubshell" "https://api.aihub.or.kr/api/aihubshell.do"
    chmod +x aihubshell
else
    echo "1. aihubshell이 이미 존재합니다."
fi

# --- 메인 작업 루프 (파일키 순회) ---
for i in "${!FILE_KEYS[@]}"; do                                   # !FILE_KEYS[@] : 배열 인덱스 목록
    key=${FILE_KEYS[$i]}
    num_for_folder=${FILE_NUMBERS[$i]}
    padded_num=$(printf "%02d" "$num_for_folder")

    echo -e "\n------------------------------------------------------------"
    echo "2. 파일키 [$padded_num] (파일키: $key) 다운로드를 시작합니다."

    ZIP_FILE="${DOWNLOAD_PATH}/${padded_num}_real_sen_keypoint.zip"
    UNZIPPED_FOLDER="${DOWNLOAD_PATH}/${padded_num}"

    cd /mnt/disks/data

    if [ -f "$ZIP_FILE" ]; then
        echo "   Already existing file '$ZIP_FILE'..."
    else
		    # 1. aihubshell로 특정 파일키 데이터 다운로드
		    echo "   aihubshell로 데이터 다운로드 중..."
		    ./aihubshell -mode d -datasetkey $DATASET_ID -filekey $key -aihubapikey $AIHUB_API_KEY
    fi

    # 다운로드된 최상위 폴더 이름 동적 찾기
    if [ ! -f "$ZIP_FILE" ]; then
        echo " XXXX 다운로드 실패! '$ZIP_FILE'을 찾을 수 없습니다. 다음 파일키로 넘어갑니다."
        continue
    fi
    echo "   -> 다운로드된 ZIP 파일: '$ZIP_FILE'"
    echo "3. ZIP 파일 압축을 해제합니다..."
    7z x "$ZIP_FILE" -o"$UNZIPPED_FOLDER" -y

    echo "4. GCS 업로드 시작 (제외 대상 필터링 중...)"

    # GCS에 업로드할 폴더 경로 구성 및 업로드
    find "$UNZIPPED_FOLDER" -maxdepth 1 -type d -name "NIA_SL_SEN*" | while read folder_path; do
        folder_name=$(basename "$folder_path")
        # 번호 4자리 추출 (예: NIA_SL_SEN0019 -> 0019)
        sen_num=$(echo "$folder_name" | grep -oE '[0-9]{4}' | head -1)

        if [[ "$EXCLUDE_LIST" =~ "$sen_num" ]]; then
            echo "  >> SKIP (보유중): $folder_name"
        else
            # 제외 목록에 없으면 GCS로 전송
            # -n 옵션은 clobber(덮어쓰기) 방지, -r은 재귀
            gcloud storage cp -r -n "$folder_path" "${GCS_BUCKET_NAME}/"
        fi
    done

    # 작업 완료 후 다운로드 받은 폴더 삭제
    echo "5. 공간 확보를 위해 다운로드한 폴더 [$ZIP_FILE] 및 [$UNZIPPED_FOLDER]를 삭제합니다."
		rm -f "$ZIP_FILE"
    rm -rf "$UNZIPPED_FOLDER"

    echo " VVVV 파일키 [$padded_num] 작업 완료."
done

echo "------------------------------------------------------------"
echo " VVVV 모든 작업이 성공적으로 완료되었습니다!"
