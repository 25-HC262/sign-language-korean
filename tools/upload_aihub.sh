#!/bin/bash

# 설치 가이드라인:
# Windows의 경우 wsl 다운로드하여 리눅스 환경에서 할 것을 추천 (Powershell 관리자 모드: `wsl --install -d Ubuntu`)
# sudo apt update
# awscli는 따로 공식 가이드라인 참고하여 설치
# sudo apt install -y p7zip-full, jq (json parser)
# aws configure 후 Access Key, Secret Access Key, region name, output format : json

# pwd ~에서 진행

AIHUB_API_KEY=""                      # AIHub에서 신청해야 함
DATASET_ID=103                        # 수어영상 데이터셋 아이디 103
S3_BUCKET_NAME="openpose-keypoints"   # 수어영상 키포인트는 "openpose-keypoints"

FILE_KEYS=(39590 39591 39592 39593 39594 39596 39597 39598 39599)  # AIHub 페이지에서 찾을 수 있음.

# 위 FILE_KEYS와 1:1로 대응되는 폴더 숫자 목록
FILE_NUMBERS=(7 8 9 10 11 13 14 15 16)                             # EX. 08_real_sen_keypoint 폴더처럼 맨앞에 붙는 PREFIX

# 수어 문장 목록의 숫자들 의미
TARGET_NUMBERS=(0019 0021 0033 0035 0067 0109 0111 0133 0145 0181 0183 0185 0187 0189 0191 0193 0195 0197 0354 0355 0356 0357 1045 1613 1771 1773 1775 1940 1945 1976 1978 1999 2000)

# aihubshell이 파일을 다운로드하는 고정된 기본 경로
DOWNLOAD_PATH="004.수어영상/1.Training/라벨링데이터/REAL/SEN"          # 수어영상 다운로드 구조 (수동 설치와 구조 상이); 압축해제할 경우 하위폴더 FILENUM이 붙는다

echo "AI Hub 자동 다운로드 및 S3 업로드 스크립트를 시작합니다."

# --- 2. aihubshell 준비 ---
if [ ! -f "aihubshell" ]; then
    echo "1. aihubshell을 다운로드합니다."
    curl -L -o "aihubshell" "https://api.aihub.or.kr/api/aihubshell.do"
    chmod +x aihubshell
else
    echo "1. aihubshell이 이미 존재합니다."
fi

# 다운로드 경로가 없으면 생성
if [ ! -d "$DOWNLOAD_PATH" ]; then
    echo "2. 기본 다운로드 경로 '$DOWNLOAD_PATH'를 생성합니다."
    mkdir -p "$DOWNLOAD_PATH"
else
    echo "2. 기본 다운로드 경로 '$DOWNLOAD_PATH'가 이미 존재합니다."
fi
echo " VVVV 사전 준비 완료."

# --- 3. 메인 작업 루프 (파일키 순회) ---
for i in "${!FILE_KEYS[@]}"; do                                   # !FILE_KEYS[@] : 배열 인덱스 목록
    key=${FILE_KEYS[$i]}
    num_for_folder=${FILE_NUMBERS[$i]}
    padded_num=$(printf "%02d" "$num_for_folder")

    echo -e "\n------------------------------------------------------------"
    echo "-> 파일키 [$key] (폴더번호: $padded_num) 작업을 시작합니다."

    ZIP_FILE="${DOWNLOAD_PATH}/${padded_num}_real_sen_keypoint.zip"
    # UNZIPPED_FOLDER="${DOWNLOAD_PATH}"

    if [ -f "$ZIP_FILE" ]; then
        echo "   1) Already existing file '$ZIP_FILE'..."
    else
		    # 1. aihubshell로 특정 파일키 데이터 다운로드
		    echo "   1) aihubshell로 데이터 다운로드 중..."
		    ./aihubshell -mode d -datasetkey $DATASET_ID -filekey $key -aihubapikey $AIHUB_API_KEY
    fi

    # 2. 다운로드된 최상위 폴더 이름 동적 찾기
    if [ ! -f "$ZIP_FILE" ]; then
        echo " XXXX 다운로드 실패! '$ZIP_FILE'을 찾을 수 없습니다. 다음 파일키로 넘어갑니다."
        continue
    fi
    echo "   -> 다운로드된 ZIP 파일: '$ZIP_FILE'"
    echo "   2) ZIP 파일 압축을 해제합니다..."
    #unzip -q -o "$ZIP_FILE" -d "$DOWNLOAD_PATH" # -q: 조용히 실행, -o: 덮어쓰기       # unzip이 느린 관계로 p7zip으로 변경
    7z x "$ZIP_FILE" -o"$DOWNLOAD_PATH" -y

    # 3. S3에 업로드할 폴더 경로 구성 및 업로드
    SEARCH_PATH="${DOWNLOAD_PATH}/${padded_num}/"
    echo "   3) 다음 경로에서 업로드 대상 폴더를 검색합니다: '$SEARCH_PATH'"
    if [ -d "$SEARCH_PATH" ]; then
        # 지정된 숫자 목록(TARGET_NUMBERS)을 순회하며 대상 폴더 검색 및 업로드
        for target_num in "${TARGET_NUMBERS[@]}"; do
            # "NIA_SL_SEN[숫자]_REAL*" 패턴에 맞는 폴더들을 찾음
            find "$SEARCH_PATH" -type d -name "NIA_SL_SEN${target_num}_REAL*" | while read source_folder; do
                if [ -d "$source_folder" ]; then
                    s3_folder_name=$(basename "$source_folder")
                    echo "    -> '${s3_folder_name}' 폴더를 s3://${S3_BUCKET_NAME}/${s3_folder_name}/ 경로로 업로드합니다."
                    aws s3 sync "$source_folder" "s3://${S3_BUCKET_NAME}/${s3_folder_name}/" --quiet
                fi
            done
        done
    else
        echo " XXXX 경고: 검색 경로를 찾을 수 없습니다. 폴더 구조를 다시 확인해주세요."
        ls -l ${UNZIPPED_FOLDER}
    fi

    # 4. 작업 완료 후 다운로드 받은 폴더 삭제
    echo "   4) 공간 확보를 위해 다운로드한 폴더 [$ZIP_FILE] 및 [$UNZIPPED_FOLDER]를 삭제합니다."
		rm -f "$ZIP_FILE"
    rm -rf "$UNZIPPED_FOLDER"

    echo " VVVV 파일키 [$key] 작업 완료."
done

echo "------------------------------------------------------------"
echo " VVVV 모든 작업이 성공적으로 완료되었습니다!"
