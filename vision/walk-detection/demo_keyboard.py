import os
import cv2
import numpy as np
import time
import math
import pygame

# ==============================================================================
# 設定変数 (CONFIGURATION)
# ==============================================================================

# --- 動画・音声ファイルパス ---
VIDEO_PATH = "wave.mp4"
AUDIO_LOW_PATH = "Layer_Low.wav"
AUDIO_MID_PATH = "Layer_Mid.wav"
AUDIO_HIGH_PATH = "Layer_High.wav"

# --- 入力・速度制御パラメータ ---
STEP_TIMEOUT = 1.5           # 打鍵が止まったと判定するまでの時間(秒)
SPEED_MULTIPLIER = 1.0       # 打鍵ピッチ(歩/秒)から目標速度(v_target)への変換係数
MAX_TARGET_SPEED = 3.0       # v_targetの最大値 (映像の最大倍速)
MIN_ACTUAL_SPEED = 0.01      # v_actualが0のときの揺らぎモードの再生速度

# --- 物理モデル(慣性・ダンパ)パラメータ ---
# 値が大きいほど目標速度への追従が速くなる(0.0 〜 無限大)
# LPF係数 k: v_actual += (v_target - v_actual) * (1 - exp(-k * dt))
INERTIA_COEFF = 2.0          

# --- 音響制御（ボリュームカーブ）パラメータ ---
# v_actual の値に応じてフェードイン・アウトする閾値
V_ACTUAL_MID_START = 0.5
V_ACTUAL_MID_FULL  = 1.0
V_ACTUAL_HIGH_START = 1.2
V_ACTUAL_HIGH_FULL  = 2.0

# ==============================================================================

def load_sound(path):
    if os.path.exists(path):
        return pygame.mixer.Sound(path)
    print(f"Warning: Audio file not found: {path} (Skipping...)")
    return None

def main():
    # 1. 音響初期化 (Pygame Mixer)
    # 事前に `pip install pygame` が必要です。
    pygame.mixer.init()
    
    snd_low = load_sound(AUDIO_LOW_PATH)
    snd_mid = load_sound(AUDIO_MID_PATH)
    snd_high = load_sound(AUDIO_HIGH_PATH)
    
    # ループ再生 (-1)
    ch_low = snd_low.play(loops=-1) if snd_low else None
    ch_mid = snd_mid.play(loops=-1) if snd_mid else None
    ch_high = snd_high.play(loops=-1) if snd_high else None

    # 初期ボリュームは0
    if ch_low: ch_low.set_volume(0.0)
    if ch_mid: ch_mid.set_volume(0.0)
    if ch_high: ch_high.set_volume(0.0)
    
    # 2. 映像初期化 (OpenCV)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Warning: Cannot open video file '{VIDEO_PATH}'.")
        print("Using blank frame as fallback...")
        fps = 30.0
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30.0

    # 状態変数
    v_actual = 0.0
    v_target = 0.0
    avg_pitch = 0.0
    
    last_step_time = 0.0
    pitch_history = []
    
    cv2.namedWindow("Keyboard Demo", cv2.WINDOW_NORMAL)
    
    print("\n" + "="*50)
    print(" [時間の渚] キーボード入力制御デモ")
    print(" - 'Enter'キーをリズミカルに押して再生速度・音響を制御")
    print(" - 'ESC'キーで終了")
    print("="*50 + "\n")

    # 最初のフレームを読み込み（フォールバック時は黒画面生成）
    ret, current_frame = cap.read() if cap.isOpened() else (True, np.zeros((720, 1280, 3), dtype=np.uint8))
    if not ret:
        print("Error: Could not read the first frame.")
        return

    last_loop_time = time.time()
    last_frame_time = time.time()

    while True:
        current_time = time.time()
        dt = current_time - last_loop_time
        last_loop_time = current_time
        
        # 3. 入力処理 (OpenCVのイベント)
        # 連続的な制御のため、waitKeyの待ち時間は最低(1ms)にする
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27: # ESCキー
            break
        elif key == 13: # Enterキー
            if last_step_time == 0.0 or (current_time - last_step_time) > STEP_TIMEOUT:
                # 最初の1歩、またはタイムアウト後の1歩（まずは標準的なピッチで始動）
                avg_pitch = 1.0 
                v_target = min(avg_pitch * SPEED_MULTIPLIER, MAX_TARGET_SPEED)
            else:
                interval = current_time - last_step_time
                if interval > 0.05: # デバウンス処理（短すぎる間隔は無視）
                    pitch = 1.0 / interval
                    pitch_history.append(pitch)
                    # 直近3件の移動平均
                    if len(pitch_history) > 3:
                        pitch_history.pop(0)
                    avg_pitch = sum(pitch_history) / len(pitch_history)
                    v_target = min(avg_pitch * SPEED_MULTIPLIER, MAX_TARGET_SPEED)
            last_step_time = current_time

        # 4. 減衰処理 (タイムアウト)
        time_since_last_step = current_time - last_step_time
        if time_since_last_step > STEP_TIMEOUT:
            v_target = 0.0
            avg_pitch = 0.0
            pitch_history.clear()

        # 5. 物理エンジン（指数移動平均 LPF）
        # 経過時間 dt に対してどれだけ追従するか (フレームレート非依存)
        alpha = 1.0 - math.exp(-INERTIA_COEFF * dt)
        v_actual += (v_target - v_actual) * alpha
        
        # 6. 音響制御 (ボリューム計算)
        # Lowは揺らぎモード(v_actual≒0)でも再生を止めない前提だが、完全停止時は半分程度に下げる
        vol_low = 1.0 if v_actual > 0.1 else 0.5 
        
        # Mid は閾値に従いフェードイン
        vol_mid = np.clip((v_actual - V_ACTUAL_MID_START) / (V_ACTUAL_MID_FULL - V_ACTUAL_MID_START + 1e-5), 0.0, 1.0)
        
        # High も閾値に従いフェードイン
        vol_high = np.clip((v_actual - V_ACTUAL_HIGH_START) / (V_ACTUAL_HIGH_FULL - V_ACTUAL_HIGH_START + 1e-5), 0.0, 1.0)
        
        if ch_low: ch_low.set_volume(vol_low)
        if ch_mid: ch_mid.set_volume(vol_mid)
        if ch_high: ch_high.set_volume(vol_high)

        # 7. 映像制御
        # 揺らぎモード用のクランプ
        v_play = max(v_actual, MIN_ACTUAL_SPEED)
        
        # 再生時のフレーム間隔を算出
        base_frame_time = 1.0 / fps
        frame_interval = base_frame_time / v_play
        
        # 指定されたインターバルが経過したら次のフレームを読み込む
        if current_time - last_frame_time >= frame_interval:
            last_frame_time = current_time
            if cap.isOpened():
                ret, next_frame = cap.read()
                # 動画が終端にきたらループ
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, next_frame = cap.read()
                if ret:
                    current_frame = next_frame.copy()

        # UI描画用のコピーを用意
        display_frame = current_frame.copy()
        
        # --- OSD (On Screen Display) の描画処理 ---
        def draw_text_with_outline(img, text, pos, color=(255, 255, 255)):
            font = cv2.FONT_HERSHEY_SIMPLEX
            # アウトライン (黒)
            cv2.putText(img, text, pos, font, 1.0, (0, 0, 0), 4)
            # 本体
            cv2.putText(img, text, pos, font, 1.0, color, 2)
            
        draw_text_with_outline(display_frame, f"v_actual: {v_actual:.3f}x", (30, 50), (0, 255, 255))
        draw_text_with_outline(display_frame, f"v_target: {v_target:.2f}x", (30, 100), (255, 255, 255))
        draw_text_with_outline(display_frame, f"Pitch:    {avg_pitch:.2f} steps/s", (30, 150), (0, 255, 0))
        draw_text_with_outline(display_frame, f"Audio [L:{vol_low:.2f} M:{vol_mid:.2f} H:{vol_high:.2f}]", (30, 200), (255, 100, 100))

        # 画面表示
        cv2.imshow("Keyboard Demo", display_frame)

    # 終了処理
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()
