import cv2
import numpy as np
import time
import math
import socket
import threading
import pygame
import os

# ==============================================================================
# 設定変数 (CONFIGURATION)
# ==============================================================================

# --- ファイルパス設定 ---
VIDEO_PATH = "wave-8x-RIFE-RIFE3.1-239.7602fps.mp4"
AUDIO_LOW_PATH = "Layer_Low.wav"   # 静かな環境音 (常に再生)
AUDIO_MID_PATH = "Layer_Mid.wav"   # 波の砕ける音など (中速以上でフェードイン)
AUDIO_HIGH_PATH = "Layer_High.wav" # 引き波の音など (高速でフェードイン)

# --- ネットワーク・UDP設定 ---
UDP_IP = "0.0.0.0"
UDP_PORT = 1902

# --- 物理モデル・速度制御定数 ---
ALPHA = 2.0  # 加速度係数 (目標速度への追従の速さ)
BETA = 1.5   # 減衰係数 (目標速度への減速の速さ)
MIN_ACTUAL_SPEED = 0.005 # v_actualが0のときの極低速（微細な揺らぎ）

# ==============================================================================

# UDP受信用グローバル変数
udp_data = {"left_lift": 0.0, "right_lift": 0.0, "total_lift": 0.0, "step_length": 0.0}
raw_udp_str = "Waiting for UDP..."
is_running = True

def udp_listener():
    """バックグラウンドでUDP信号を受信するスレッド"""
    global udp_data, raw_udp_str, is_running
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(0.5)
    
    while is_running:
        try:
            data, addr = sock.recvfrom(1024)
            data_str = data.decode('utf-8').strip()
            raw_udp_str = data_str
            parts = data_str.split(',')
            if len(parts) >= 4:
                udp_data["left_lift"] = float(parts[0].strip())
                udp_data["right_lift"] = float(parts[1].strip())
                udp_data["total_lift"] = float(parts[2].strip())
                udp_data["step_length"] = float(parts[3].strip())
        except socket.timeout:
            pass
        except Exception as e:
            raw_udp_str = f"Error: {e}"

def load_sound(path):
    if os.path.exists(path):
        return pygame.mixer.Sound(path)
    print(f"Warning: Audio file not found: {path} (Skipping...)")
    return None

def on_trackbar(val):
    # Trackbarのイベントハンドラ(ここでは何もしない)
    pass

class VideoReaderThread:
    """映像の読み込みとデコードを別スレッドで行うクラス"""
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"Warning: Cannot open video file '{video_path}'.")
            self.fps = 30.0
            self.total_frames = 1000
        else:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0: self.fps = 30.0
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.total_frames <= 0: self.total_frames = 1000000

        self.target_vframe = 0
        self.current_vframe = 0
        self.latest_frame = None
        self.lock = threading.Lock()
        
        # 初期フレームの読み取り
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = frame
                self.current_vframe = 0

        self.is_running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def update_target(self, vframe):
        """メインスレッドから目標のフレームインデックスを通知する"""
        self.target_vframe = vframe

    def get_frame(self):
        """常に最新(目標時刻に一番近い)のフレームを返す"""
        with self.lock:
            return self.latest_frame

    def _read_frame_with_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return ret, frame

    def _grab_with_loop(self):
        ret = self.cap.grab()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret = self.cap.grab()
        return ret

    def _update_loop(self):
        """目標フレームへの追いつき処理 (高速なスキップと読み込み)"""
        while self.is_running:
            target = self.target_vframe
            diff = target - self.current_vframe

            if diff > 0 and self.cap.isOpened():
                if diff > self.total_frames:
                    # 目標が動画全体長より大きく飛んでいる場合はset()でシーク
                    jump_pos = target % self.total_frames
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, jump_pos)
                    ret, frame = self._read_frame_with_loop()
                    if ret:
                        with self.lock:
                            self.latest_frame = frame
                    self.current_vframe = target
                else:
                    # 数フレームのスキップは grab() を連続利用してデコード負荷を削減
                    for _ in range(diff - 1):
                        self._grab_with_loop()
                        self.current_vframe += 1
                    
                    # 最後の1フレームは read() で画像を取得
                    ret, frame = self._read_frame_with_loop()
                    if ret:
                        with self.lock:
                            self.latest_frame = frame
                    self.current_vframe += 1
            else:
                # ターゲットに追いついている場合は少し待機してCPU負荷軽減
                time.sleep(0.001)

    def release(self):
        self.is_running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()

def main():
    global is_running
    
    # 1. UDPスレッド開始
    threading.Thread(target=udp_listener, daemon=True).start()

    # 2. 音響初期化
    pygame.mixer.init()
    snd_low = load_sound(AUDIO_LOW_PATH)
    snd_mid = load_sound(AUDIO_MID_PATH)
    snd_high = load_sound(AUDIO_HIGH_PATH)

    ch_low = snd_low.play(loops=-1) if snd_low else None
    ch_mid = snd_mid.play(loops=-1) if snd_mid else None
    ch_high = snd_high.play(loops=-1) if snd_high else None

    if ch_low: ch_low.set_volume(0.0)
    if ch_mid: ch_mid.set_volume(0.0)
    if ch_high: ch_high.set_volume(0.0)

    # 3. 映像初期化 (専用スレッドを使用)
    reader = VideoReaderThread(VIDEO_PATH)
    fps = reader.fps

    cv2.namedWindow("Inertia Engine UI", cv2.WINDOW_NORMAL)
    
    # --- Trackbars等のUI初期化 ---
    # Mode Selector: 0=Auto(UDP), 1=Manual(Trackbar)
    cv2.createTrackbar("Mode(0:A 1:M)", "Inertia Engine UI", 0, 1, on_trackbar)
    # Debug Speed: 0-100 (マッピング後 0.0-5.0)
    cv2.createTrackbar("Debug Speed", "Inertia Engine UI", 0, 100, on_trackbar)
    # Mapping Factor: 0-100 (マッピング後 0.0-5.0)
    cv2.createTrackbar("Mapping Factor", "Inertia Engine UI", 20, 100, on_trackbar)
    # Inertia Intensity: 0-100 (マッピング後 0.0-2.0倍率)
    cv2.createTrackbar("Inertia", "Inertia Engine UI", 50, 100, on_trackbar)

    v_actual = 0.0
    virtual_video_time = 0.0
    
    last_loop_time = time.perf_counter()
    last_print_time = time.perf_counter()
    show_osd = False  # デフォルトは非表示(ターミナル出力)
    
    # 最初のダミーフレーム
    current_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    print("\n" + "="*50)
    print(" [Inertia Engine] 起動完了 (240fps/高精度再生モード)")
    print(" - UDPポート: 1902 で待機中")
    print(" - UI上で動作モード(Auto/Manual)を切り替え可能")
    print(" - [O]キーで画面上テキスト(OSD)の表示/非表示をトグル")
    print(" - 終了するにはESCキーを押してください")
    print("="*50 + "\n")

    while True:
        current_time = time.perf_counter()
        dt = current_time - last_loop_time
        last_loop_time = current_time
        
        # --- UIから値取得 ---
        mode = cv2.getTrackbarPos("Mode(0:A 1:M)", "Inertia Engine UI")
        debug_speed_val = cv2.getTrackbarPos("Debug Speed", "Inertia Engine UI")
        mapping_val = cv2.getTrackbarPos("Mapping Factor", "Inertia Engine UI")
        inertia_val = cv2.getTrackbarPos("Inertia", "Inertia Engine UI")

        # --- 1. 入力値(v_input)の決定 ---
        if mode == 1:
            # Manual Mode: スライダーの値を0.0〜5.0に変換して使用
            v_input = debug_speed_val / 20.0
        else:
            # Auto Mode: UDPで受信した total_lift を使用
            v_input = float(udp_data["total_lift"])

        # --- 2. 目標速度(v_target)の算出 ---
        mapping_factor = mapping_val / 20.0  # デフォルト=1.0 (20/20.0)
        v_target = v_input * mapping_factor
        
        # --- 3. 物理エンジン(Inertia Model) ---
        # スライダーから慣性の強さの倍率を取得（50が基準の1.0倍）
        inertia_scale = inertia_val / 50.0  
        current_alpha = ALPHA * inertia_scale
        current_beta = BETA * inertia_scale
        
        # v_actualがv_targetに向かう際、加速か減速かで定数を切り替え
        coeff = current_alpha if v_target >= v_actual else current_beta
        
        # ユーザ指定の慣性モデル: v_actual = v_actual_prev + coeff * (v_target - v_actual_prev)
        # dtによる指数緩和を用いてフレームレート非依存にする
        rate = 1.0 - math.exp(-coeff * dt)
        v_actual += rate * (v_target - v_actual)
        
        # --- 4. 多層音響制御 (クロスフェード) ---
        # 静止時(0)〜極低速時は静かな環境音(Low)のみ
        vol_low = 1.0 if v_actual > 0.05 else 0.5
        
        # 中速域で波の音(Mid)をフェードイン (v_actual=0.5〜1.0で0.0->1.0に)
        vol_mid = np.clip((v_actual - 0.5) / 0.5, 0.0, 1.0)
        
        # 高速域で引き波の音(High)をフェードイン (v_actual=1.2〜2.0で0.0->1.0に)
        vol_high = np.clip((v_actual - 1.2) / 0.8, 0.0, 1.0)
        
        if ch_low: ch_low.set_volume(vol_low)
        if ch_mid: ch_mid.set_volume(vol_mid)
        if ch_high: ch_high.set_volume(vol_high)
        
        # --- 5. 映像制御 ---
        # 完全停止を避け、MIN_ACTUAL_SPEED（極低速）を維持する
        v_play = max(v_actual, MIN_ACTUAL_SPEED)
        
        # 経過時間から本来表示されるべき動画上の時刻(フレーム)を計算
        virtual_video_time += dt * v_play
        target_vframe = int(virtual_video_time * fps)
        
        # スレッドに目標フレームを通知して、最新バッファを取得
        reader.update_target(target_vframe)
        f = reader.get_frame()
        if f is not None:
            current_frame = f
            
        # --- 描画 (OSD: On Screen Display) ---
        if show_osd:
            display_frame = current_frame.copy()
            
            def draw_text(img, text, pos, color):
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, text, pos, font, 0.7, (0,0,0), 3)
                cv2.putText(img, text, pos, font, 0.7, color, 2)
                
            mode_text = "AUTO(UDP)" if mode == 0 else "MANUAL(Slider)"
            draw_text(display_frame, f"Mode: {mode_text}", (30, 40), (200, 200, 255))
            draw_text(display_frame, f"UDP: {raw_udp_str}", (30, 80), (255, 200, 200))
            draw_text(display_frame, f"v_in: {v_input:.2f} | target: {v_target:.2f} | actual: {v_actual:.3f}x", (30, 120), (0, 255, 255))
            draw_text(display_frame, f"Audio [Low:{vol_low:.2f} Mid:{vol_mid:.2f} High:{vol_high:.2f}]", (30, 160), (255, 200, 100))
        else:
            # OSD無効時はコピーも省き最速化
            display_frame = current_frame
            # ターミナルへ定期的にプリント（約0.1秒間隔）
            if current_time - last_print_time > 0.1:
                m_str = "AUTO" if mode == 0 else "MAN"
                print(f"Mode={m_str} | in={v_input:.2f} | actual={v_actual:.3f}x | UDP={raw_udp_str: <20}", end='\r')
                last_print_time = current_time

        cv2.imshow("Inertia Engine UI", display_frame)
        
        # 1ms待機しつつキー入力受付
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESCキー
            print("\nExiting...")
            break
        elif key == ord('o') or key == ord('O'):
            show_osd = not show_osd
            print(f"\nOSD Display: {'ON' if show_osd else 'OFF'}                         ")

        # --- 厳密なFPS管理 (busy wait) ---
        # OpenCVのwaitKeyだけではOSのタイマー精度(最大15ms)によりブレるため、
        # time.perf_counter()を使用して目標のフレーム描画タイミングまで待機する
        target_interval = 1.0 / (fps * v_play)
        ideal_next_time = last_loop_time + target_interval
        
        while time.perf_counter() < ideal_next_time:
            pass

    # --- 終了処理 ---
    is_running = False
    reader.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()
