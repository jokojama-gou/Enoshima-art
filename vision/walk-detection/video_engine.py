import cv2
import socket
import threading
import time
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
UDP_IP = "127.0.0.1"
UDP_PORT = 1902
VIDEO_PATH = "./wave.mp4"

PLAYBACK_SPEED = 0.5      # 動画の再生速度（1.0で等倍。波が速いとのことなので0.5に半減して重厚感を出す）
TOTAL_DURATION = 12.0     # 歩行が止まってから完全に波が静止するまでの時間（Enter1回で波がたっぷり動くよう延長）
FADE_DURATION = 7.0       # 波がフリーズしていく尺（よりゆっくりとフェードさせる）
FADE_START = TOTAL_DURATION - FADE_DURATION  # フェード開始時刻 (5.0秒)

TARGET_FPS = 60
FRAME_TIME = 1.0 / TARGET_FPS

class VisualEngine:
    """
    UDPシグナルをトリガーとして、動画再生とフェード処理を行うビジュアルエンジン。
    【アート作品仕様】
    歩行検知（トリガー）が続いている間は波が延々とループ再生され、
    トリガーが途絶えると、徐々に波が静止（フリーズ）していく演出を行う。
    """
    def __init__(self):
        self.frames = []
        self.is_running = True
        
        # 再生ステータス
        self.last_trigger_time = 0
        self.is_active = False
        self.current_frame_idx = 0.0
        
        # ウィンドウ設定
        self.window_name = "Art Installation - Visual Engine"
        self._setup_window()
        self._preload_video()
        
        # UDP設定
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))
        self.sock.settimeout(0.5)

    def _setup_window(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def _preload_video(self):
        print(f"[{VIDEO_PATH}] をメモリにプリロードしています...")
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"Error: 動画ファイル '{VIDEO_PATH}' が見つからないか開けません。")
            self.frames = [np.zeros((720, 1280, 3), dtype=np.uint8)]
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
        cap.release()
        print(f"ロード完了: 計 {len(self.frames)} フレームを読み込みました。")

    def _udp_listener_thread(self):
        print(f"UDPリスナー待機中: {UDP_IP}:{UDP_PORT}")
        while self.is_running:
            try:
                data, addr = self.sock.recvfrom(1024)
                # ログが増えすぎるのを防ぐため、デバッグ以外ならprintを消すか適宜調整する
                # print(f"[UDP Trigger] {addr} からシグナルを受信: 歩行検知")
                self.trigger_playback()
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    print(f"UDP Listener Error: {e}")

    def trigger_playback(self):
        """動画を進行させる。連続トリガーによって時間を延長し、途切れないようにする"""
        self.last_trigger_time = time.time()
        self.is_active = True

    def run(self):
        udp_thread = threading.Thread(target=self._udp_listener_thread, daemon=True)
        udp_thread.start()

        # 波が止まっている時（フリーズ）用のフレーム。最初は動画の0フレーム目。
        idle_frame = self.frames[0] if self.frames else np.zeros((720, 1280, 3), dtype=np.uint8)

        print("起動完了。")
        print(" - 作品用モード: 連続トリガーで波が進み、途絶えるとゆっくり波が静止します")
        print(" - [Enter] キーでデバッグトリガーを発行（長押しで歩きへの疑似テスト可能）")
        print(" - [Esc] キーで終了します")

        while self.is_running:
            loop_start = time.time()
            
            if self.is_active:
                # 最後のトリガー（歩行入力）からの経過時間を計算
                time_since_trigger = time.time() - self.last_trigger_time
                
                if time_since_trigger < TOTAL_DURATION:
                    # トリガーが生きている（＝人がいる）限り、速度に応じて進む（ループ）
                    self.current_frame_idx = (self.current_frame_idx + PLAYBACK_SPEED) % len(self.frames)
                
                raw_frame = self.frames[int(self.current_frame_idx)]
                
                # --- 歩行停止から完全に波が静止するまでのフェードアルゴリズム ---
                if time_since_trigger < FADE_START:
                    # 最後に歩いてから一定時間以内なら、そのまま動く
                    opacity = 1.0
                elif time_since_trigger <= TOTAL_DURATION:
                    # 歩き止まってから時間が経つと、動きが重くなっていく
                    opacity = 1.0 - (time_since_trigger - FADE_START) / FADE_DURATION
                    opacity = max(0.0, opacity)
                else:
                    # 完全に停止
                    opacity = 0.0
                    self.is_active = False 

                # 描写ロジック
                if opacity >= 1.0:
                    display_frame = raw_frame
                    # 波が元気に動いている時の最新フレームを「停止目標」として記憶し続ける
                    # これにより、歩き止まった瞬間の波の形が固定（フリーズ）される
                    idle_frame = raw_frame 
                elif opacity <= 0.0:
                    display_frame = idle_frame
                else:
                    # 動いている波(raw_frame) と 止まっている波(idle_frame) を混ぜる
                    # 波が次第に静止していくモーションブラーのような視覚効果を生む
                    display_frame = cv2.addWeighted(raw_frame, opacity, idle_frame, 1.0 - opacity, 0)
            else:
                # 完全な待機状態
                display_frame = idle_frame

            cv2.imshow(self.window_name, display_frame)

            # --- フレームレート維持 ---
            process_time = time.time() - loop_start
            sleep_time_ms = max(1, int((FRAME_TIME - process_time) * 1000))
            
            key = cv2.waitKey(sleep_time_ms) & 0xFF
            
            if key == 27:
                self.is_running = False
            elif key == 13: # Enter
                self.trigger_playback()

        self.is_running = False
        self.sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    engine = VisualEngine()
    engine.run()
