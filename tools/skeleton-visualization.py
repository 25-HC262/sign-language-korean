import json
import os.path

import cv2
import numpy as np
import imageio
from train.gloss_transformer_train import get_dataset

'''1회용 gif 만드는 코드, 폴더 구조는 data/train 내에 문장 폴더 -> 방향별 폴더 -> 개인폴더 여야 함'''

# 1) COCO-18 body skeleton (OpenPose pose_keypoints_2d 기준)
POSE_PAIRS = [
    (0,1),(1,2),(2,3),(3,4),      # nose→neck→Rsho→Relb→Rwri
    (1,5),(5,6),(6,7),            # neck→Lsho→Lelb→Lwri
    (1,8),(8,9),(9,10),           # neck→MidHip→RHip→RKnee
    (8,11),(11,12),(12,13)        # midHip→LHip→LKnee→LAnk
]

# 2) Hand skeleton (21 pts)
HAND_PAIRS = [(i, i+1) for i in range(0, 20)]  # wrist(0)→thumb(1→4), index(5→8), middle(9→12), ring(13→16), pinky(17→20)

# 3) Face “mesh” is large – 여기서는 단순히 랜덤으로 인접한 몇 개만 연결해 예시로 씁니다.
FACE_PAIRS = [(i, i+1) for i in range(0,69)]  # 사실은 Mediapipe FaceMesh 순서대로 연결 필요

def draw_from_json(json_str, canvas=None):
    if isinstance(json_str, str):
        data = json.loads(json_str)
    ppl = data.get("people", {})
    person = ppl if isinstance(ppl, dict) else (ppl[0] if ppl else {})

    # JSON 에서 2D keypoints 뽑기
    def kv(name, n):
        arr = person.get(name, [])
        pts = np.array(arr).reshape((-1,3)) if len(arr) == n*3 else np.zeros((n,3))
        return pts

    pose = kv("pose_keypoints_2d", 25)
    lhand= kv("hand_left_keypoints_2d", 21)
    rhand= kv("hand_right_keypoints_2d",21)
    face = kv("face_keypoints_2d", 70)

    # 빈 캔버스 생성
    if canvas is None:
        # 얼굴 좌표 중심으로 캔버스 크기 잡기 (혹은 고정 크기)
        canvas = np.zeros((720,1280,3), dtype=np.uint8)

    # 함수: (x,y,c)->(int(x),int(y)) 픽셀
    def to_pt(p): return (int(p[0]), int(p[1]))

    # 1) body 그리기
    for i,j in POSE_PAIRS:
        if pose[i,2]>0.1 and pose[j,2]>0.1:
            cv2.line(canvas, to_pt(pose[i]), to_pt(pose[j]), (0,255,0), 2)
    for i in range(len(pose)):
        if pose[i,2]>0.1:
            cv2.circle(canvas, to_pt(pose[i]), 4, (0,0,255), -1)

    for pts, color in [(lhand,(255,0,0)), (rhand,(255,255,0))]:
        for i,j in HAND_PAIRS:
            if pts[i,2]>0.1 and pts[j,2]>0.1:
                cv2.line(canvas, to_pt(pts[i]), to_pt(pts[j]), color, 2)
    for i in range(len(pts)):
        if pts[i,2]>0.1:
            cv2.circle(canvas, to_pt(pts[i]), 3, color, -1)

    # 3) face 그리기 (단순 예시)
    for i,j in FACE_PAIRS:
        if face[i,2]>0.1 and face[j,2]>0.1:
            cv2.line(canvas, to_pt(face[i]), to_pt(face[j]), (0,128,255), 1)
    for i in range(len(face)):
        if face[i,2]>0.1:
            cv2.circle(canvas, to_pt(face[i]), 2, (0,128,255), -1)

    return canvas

if __name__ == "__main__":
    # JSON 파일 읽어서
    pprefix = "data/train"
    for sen in os.listdir(pprefix): # sen = NIA_SL_SEN0000
        bigpath = os.path.join(pprefix, sen)
        if os.path.isdir(bigpath) and sen != "train" and sen != "val":
            for angle in os.listdir(bigpath): # angle = NIA_SL_SEN0000_D
                bbigpath = os.path.join(bigpath, angle)
                for real in os.listdir(bbigpath): # real =  NIA_SL_SEN0000_REAL01_D
                    if os.path.isdir(os.path.join(bbigpath, real)):
                        frames = []
                        kppath = os.path.join(bbigpath, real)
                        for kpfile in os.listdir(kppath):
                            if os.path.isfile(os.path.join(kppath,kpfile)) and kpfile.endswith(".json"):
                                json_path = os.path.join(kppath, kpfile)
                                with open(json_path) as f:
                                    js = f.read()
                                    img = draw_from_json(js)          # 빈 캔버스에 그리기
                                    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                        out_dir = f"openpose_out/{sen}/{angle}"
                        os.makedirs(out_dir, exist_ok=True)

                        gif_path = os.path.join(out_dir, f"{real}.gif")
                        imageio.mimsave(gif_path, frames, fps=10)
                        print(f"Saved animation to {gif_path}")

