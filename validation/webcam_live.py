"""
로컬 웹캠으로 수어 인식 실시간 테스트.
웹캠 프레임을 모델 서버로 직접 전송하고, 예측 결과를 OpenCV 창에 표시합니다.

의존성 (로컬 venv 기준):
    pip install websockets opencv-python

Usage:
    python validation/webcam_live.py
    python validation/webcam_live.py --user-id test_user --fps 10
    python validation/webcam_live.py --server ws://localhost:8000/ws
"""

import argparse
import asyncio
import json
import queue
import struct
import threading

import cv2

DEFAULT_SERVER = "wss://model.trout-model.kro.kr/ws"
DEFAULT_USER_ID = "webcam_live_test"
DEFAULT_FPS = 10


def pack_frame(user_id: str, jpeg_bytes: bytes) -> bytes:
    """모델 서버 프로토콜: [4-byte big-endian userId 길이][userId][JPEG]"""
    uid = user_id.encode("utf-8")
    return struct.pack(">I", len(uid)) + uid + jpeg_bytes


class WsWorker:
    """백그라운드 스레드에서 WebSocket을 유지하며 프레임 전송 및 응답 수신."""

    def __init__(self, server_url: str, user_id: str):
        self.server_url = server_url
        self.user_id = user_id
        self._send_queue: queue.Queue = queue.Queue(maxsize=2)
        self._last_prediction = ""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def send(self, jpeg_bytes: bytes):
        """비동기 전송 (큐 가득 차면 최신 프레임으로 교체)."""
        try:
            self._send_queue.put_nowait(jpeg_bytes)
        except queue.Full:
            try:
                self._send_queue.get_nowait()
            except queue.Empty:
                pass
            self._send_queue.put_nowait(jpeg_bytes)

    @property
    def last_prediction(self) -> str:
        return self._last_prediction

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_loop())

    async def _connect_loop(self):
        import websockets  # type: ignore

        while True:
            try:
                async with websockets.connect(self.server_url) as ws:
                    print(f"[WS] 연결됨: {self.server_url}")
                    await asyncio.gather(
                        self._sender(ws),
                        self._receiver(ws),
                    )
            except Exception as e:
                print(f"[WS] 연결 오류: {e}. 5초 후 재시도...")
                await asyncio.sleep(5)

    async def _sender(self, ws):
        while True:
            try:
                jpeg_bytes = await self._loop.run_in_executor(
                    None, lambda: self._send_queue.get(timeout=1)
                )
                payload = pack_frame(self.user_id, jpeg_bytes)
                await ws.send(payload)
            except queue.Empty:
                continue
            except Exception:
                break

    async def _receiver(self, ws):
        async for message in ws:
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                    if data.get("type") == "model_response":
                        text = data.get("text", "")
                        if text:
                            self._last_prediction = text
                            print(f"[예측] {text}")
                except Exception:
                    pass


def run(server_url: str, user_id: str, fps: int):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    worker = WsWorker(server_url, user_id)
    worker.start()

    interval_ms = int(1000 / fps)
    prev_tick = 0

    print("'q' 키로 종료합니다.")
    print(f"서버: {server_url}  |  user_id: {user_id}  |  {fps}fps")
    print(f"브라우저 확인: {server_url.replace('ws', 'http').replace('/ws', '')}/debug/stream/{user_id}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tick = cv2.getTickCount()
        elapsed_ms = (tick - prev_tick) / cv2.getTickFrequency() * 1000
        if elapsed_ms >= interval_ms:
            prev_tick = tick
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                worker.send(buf.tobytes())

        # 예측 결과 오버레이
        display = frame.copy()
        pred = worker.last_prediction
        if pred:
            cv2.rectangle(display, (0, 0), (display.shape[1], 56), (0, 0, 0), -1)
            cv2.putText(display, pred, (12, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 80), 2, cv2.LINE_AA)
        cv2.putText(display, f"id:{user_id}  {fps}fps", (8, display.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        cv2.imshow("KSL Live", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--user-id", default=DEFAULT_USER_ID)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    args = parser.parse_args()
    run(args.server, args.user_id, args.fps)
