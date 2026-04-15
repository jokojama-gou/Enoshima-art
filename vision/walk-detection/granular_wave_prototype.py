import pygame
import numpy as np
import random
import time
import os
import wave
import math
import sys
import socket
import threading
import tkinter as tk
from tkinter import messagebox

# --- Configuration ---
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 400
SAMPLE_RATE = 44100
CHANNELS = 2
NUM_LAYERS = 8      # 並列再生するレイヤー数
FADE_SEC = 0.5      # 各レイヤーのつなぎ目(ループ境界)のフェード処理

# コントロールパネルから調整されるパラメータ
GLOBAL_VOL = 0.5    # マスターボリューム係数（音割れ防止）
BASE_RIPPLE_VOL = 0.08 # スピーカーの電源落ち(プツッというノイズ)防止＆常時水流感のための最低音量
SMOOTHING_FACTOR = 0.015 # walk_speedのスムージング係数
VOL_CHANGE_SPEED = 0.05  # ボリューム変化速度
POWER_FACTOR = 2.5       # 音量カーブのべき乗係数

SOURCE_FILE = 'source1.wav'

# --- Variables ---
walk_speed = 0.0
smoothed_walk_speed = 0.0
is_running = True
layers_info = []    # 各レイヤーのチャンネルや現在のボリュームなどの情報を保持

# --- ネットワーク・UDP設定 ---
UDP_IP = "0.0.0.0"
UDP_PORT = 1902
udp_raw_data = ""
udp_walk_speed = 0.0

def udp_listener():
    """バックグラウンドでUDP信号(total_lift等)を受信するスレッド"""
    global udp_raw_data, udp_walk_speed, is_running
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(0.5)
    
    while is_running:
        try:
            data, addr = sock.recvfrom(1024)
            data_str = data.decode('utf-8').strip()
            udp_raw_data = data_str
            parts = data_str.split(',')
            if len(parts) >= 4:
                # part[2] (total_lift) を受け取り、walk_speedに見立てる
                udp_walk_speed = float(parts[2].strip())
        except socket.timeout:
            pass
        except Exception as e:
            pass

def control_panel():
    """バックグラウンドでパラメータをリアルタイム調整するGUIスレッド"""
    root = tk.Tk()
    root.title("Control Panel")
    root.geometry("350x300")
    
    def make_scale(parent, label, from_, to, resolution, var_name, initial_val):
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(frame, text=label, width=15, anchor="w").pack(side=tk.LEFT)
        scale = tk.Scale(frame, from_=from_, to=to, resolution=resolution, orient=tk.HORIZONTAL)
        scale.set(initial_val)
        
        def on_change(val):
            globals()[var_name] = float(val)
            
        scale.config(command=on_change)
        scale.pack(side=tk.RIGHT, expand=True, fill=tk.X)
    
    make_scale(root, "GLOBAL VOL", 0.0, 1.0, 0.01, "GLOBAL_VOL", GLOBAL_VOL)
    make_scale(root, "BASE RIPPLE", 0.0, 0.5, 0.01, "BASE_RIPPLE_VOL", BASE_RIPPLE_VOL)
    make_scale(root, "SMOOTHING", 0.001, 0.1, 0.001, "SMOOTHING_FACTOR", SMOOTHING_FACTOR)
    make_scale(root, "VOL CHANGE", 0.01, 0.2, 0.01, "VOL_CHANGE_SPEED", VOL_CHANGE_SPEED)
    make_scale(root, "POWER FACTOR", 1.0, 5.0, 0.1, "POWER_FACTOR", POWER_FACTOR)
    
    root.mainloop()

def ask_startup_mode():
    """起動時にTkinterダイアログで動作モードを選択させる"""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    
    ans = messagebox.askyesno(
        "Operation Mode / 動作モードの選択",
        "本番モード (UDP通信) で起動しますか？\n\n"
        "[ はい ] 本番モード : UDPポート1902から歩行データを受信します\n"
        "[ いいえ ] デバッグモード : 今まで通りマウスやキーボードで操作します",
        parent=root
    )
    root.destroy()
    return 0 if ans else 1  # 0=Auto(UDP), 1=Manual(Debug)

def generate_dummy_wave(filename):
    """
    source1.wavが存在しない場合に、テスト用の波の音（ピンクノイズ風）を生成する関数。
    """
    print(f"[{filename}] が見つからないため、ダミーの波の音を生成します...")
    length_sec = 5.0
    num_samples = int(SAMPLE_RATE * length_sec)
    
    noise = np.random.normal(0, 0.2, num_samples)
    t = np.linspace(0, length_sec, num_samples)
    envelope = (np.sin(2 * np.pi * 0.2 * t) + 1.0) / 2.0 
    
    signal = noise * envelope * 32767 * 0.8
    signal = signal.astype(np.int16)
    
    stereo_signal = np.column_stack((signal, signal))
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(stereo_signal.tobytes())
    print(f"[{filename}] を生成しました。")

def load_source_audio(filename):
    """ソース音源の読み込み"""
    if not os.path.exists(filename):
        generate_dummy_wave(filename)
        
    with wave.open(filename, 'rb') as wf:
        sampwidth = wf.getsampwidth()
        n_channels = wf.getnchannels()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        data = wf.readframes(n_frames)
        
        if sampwidth == 2:
            arr = np.frombuffer(data, dtype=np.int16)
        else:
            raise ValueError("16ビットPCMのWAVファイルを使用してください。")
            
        if n_channels == 1:
            arr = np.column_stack((arr, arr))
        elif n_channels > 2:
            arr = arr.reshape(-1, n_channels)[:, :2]
        else:
            arr = arr.reshape(-1, 2)
            
        print(f"音源をロードしました: {filename} ({framerate}Hz, {n_channels}ch, {n_frames}frames)")
        return arr

def apply_lpf_fast(arr, window_size):
    """単純な移動平均による高速ローパスフィルタ（音を籠もらせる）"""
    if window_size <= 1:
        return arr
    cumsum = np.cumsum(arr.astype(np.float32), axis=0)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    res = cumsum[window_size - 1:] / window_size
    
    # 削られた長さをゼロパディング等で同じ長さに戻す
    pad_len = arr.shape[0] - res.shape[0]
    if pad_len > 0:
        pad = np.zeros((pad_len, arr.shape[1]), dtype=np.float32)
        res = np.vstack((res, pad))
    return res.astype(np.int16)

def resample_array(arr, speed_factor):
    """ピッチ・スピードを変更するリサンプリング処理"""
    if speed_factor == 1.0:
        return arr
    
    n_frames = arr.shape[0]
    n_channels = arr.shape[1]
    
    orig_t = np.arange(n_frames)
    new_t = np.arange(0, n_frames - 1, speed_factor)
    
    new_arr = np.zeros((len(new_t), n_channels), dtype=np.float32)
    for c in range(n_channels):
        new_arr[:, c] = np.interp(new_t, orig_t, arr[:, c])
        
    return new_arr.astype(arr.dtype)

def process_layer(arr, pitch, lpf_window):
    """一つの波をベースに、動的な揺らぎ(別レイヤー)を作り出す"""
    # 1. ローパスフィルタ（LPF）で籠もり具合を調整
    processed = apply_lpf_fast(arr, lpf_window)
    
    # 2. リサンプリングによるピッチ・速度変更
    processed = resample_array(processed, pitch)
    
    # 3. 再生ループつなぎ目でのプツプツ音(クリックノイズ)を防ぐため、両端をフェードにする
    fade_len = int(SAMPLE_RATE * FADE_SEC)
    if fade_len * 2 > len(processed):
        fade_len = len(processed) // 2
        
    fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)[:, np.newaxis]
    fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)[:, np.newaxis]
    
    processed_float = processed.astype(np.float32)
    processed_float[:fade_len] *= fade_in
    processed_float[-fade_len:] *= fade_out
    processed = processed_float.astype(np.int16)
    
    # 4. レイヤー同士が完全にシンクロして不自然に増幅するのを防ぐため、ランダムに開始位置をずらす(Shift)
    # 端っこはフェードで0になっているのでrollさせても切れ目が発生しない
    shift_amount = random.randint(0, len(processed))
    processed = np.roll(processed, shift_amount, axis=0)
    
    return processed

def init_audio_layers():
    """起動時に一度だけ、各レイヤーの波形を生成・配置する"""
    global layers_info
    base_array = load_source_audio(SOURCE_FILE)
    
    print("レイヤー生成・分析中（CPU負荷の高い処理です）...")
    for i in range(NUM_LAYERS):
        # i=0(低速域)はピッチ低め、籠もり最大
        # i=7(高速域)は元のピッチ(または微アップ)、クリアな音質
        pitch = 0.85 + (0.3 / max(1, (NUM_LAYERS - 1))) * i
        lpf_window = int(30 - (29 / max(1, (NUM_LAYERS - 1))) * i)
        
        print(f"Layer {i}: Pitch={pitch:.2f}, LPF_Window={lpf_window} -> ", end="", flush=True)
        
        processed_arr = process_layer(base_array, pitch, lpf_window)
        sound = pygame.sndarray.make_sound(processed_arr)
        
        # 確実に独立したチャンネルを取得（0番〜）
        channel = pygame.mixer.Channel(i)
        
        # 無音によるスピーカーの電源落ち（復帰時のプツッというノイズ）を防ぐため、
        # レイヤー0だけは最初から最低音量を持たせる
        initial_vol = BASE_RIPPLE_VOL if i == 0 else 0.0
        
        # 永遠にループ再生開始
        channel.set_volume(initial_vol * GLOBAL_VOL)
        channel.play(sound, loops=-1)
        
        layers_info.append({
            'channel': channel,
            'target_vol': initial_vol,
            'current_vol': initial_vol
        })
        print("Done")
    print("\n初期化完了。メインループを開始します。")

def main():
    global walk_speed, smoothed_walk_speed, is_running
    
    # 起動モードの確認とUDPスレッドの立ち上げ
    operation_mode = ask_startup_mode()
    if operation_mode == 0:
        threading.Thread(target=udp_listener, daemon=True).start()
        
    # コントロールパネルの立ち上げ (両モード共通)
    threading.Thread(target=control_panel, daemon=True).start()
        
    pygame.mixer.pre_init(frequency=SAMPLE_RATE, size=-16, channels=CHANNELS, buffer=1024)
    pygame.init()
    pygame.mixer.init()
    
    # 指定数のチャンネルを確保
    pygame.mixer.set_num_channels(NUM_LAYERS + 2)
    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Parallel Crossfade Wave Engine")
    font = pygame.font.SysFont(None, 36)
    
    # 全レイヤー読み込み・再生（無音）開始
    init_audio_layers()
    
    clock = pygame.time.Clock()
    
    while is_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False
                
        # --- 入力処理 ---
        if operation_mode == 1:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_RIGHT] or keys[pygame.K_UP]:
                walk_speed = min(1.0, walk_speed + 0.02)
            elif keys[pygame.K_LEFT] or keys[pygame.K_DOWN]:
                walk_speed = max(0.0, walk_speed - 0.02)
            else:
                if pygame.mouse.get_focused():
                    mouse_x, _ = pygame.mouse.get_pos()
                    walk_speed = np.clip(mouse_x / WINDOW_WIDTH, 0.0, 1.0)
        else:
            # UDPからの受信データに基づく (Auto)
            walk_speed = np.clip(udp_walk_speed, 0.0, 1.0)
                
        # --- walk_speed自体のスムージング（慣性の強化）---
        # 急な操作でも「ふわーっ」と遅れてついてくるように、ローパスフィルタを適用
        smoothed_walk_speed += (walk_speed - smoothed_walk_speed) * SMOOTHING_FACTOR
                
        # --- 動的ボリューム制御ロジック ---
        num_layers = len(layers_info)
        
        for i in range(num_layers):
            # レイヤー配分の最適化: もっと広く重なり合うように（オーバーラップを深く）
            # activation_end を伸ばして、より広い範囲で同時に音が鳴るようにする
            activation_start = (i / num_layers) * 0.5
            activation_end = min(1.0, activation_start + 0.6)
            
            # 各レイヤーの最大ボリューム（上位レイヤーほど減衰させて音割れを防ぐ）
            max_vol = 1.0 / math.sqrt(i + 1)
            
            if smoothed_walk_speed <= activation_start and i > 0:
                target_vol = 0.0
            elif smoothed_walk_speed >= activation_end:
                target_vol = max_vol
            else:
                # フェードイン途中
                progress = (smoothed_walk_speed - activation_start) / (activation_end - activation_start)
                
                # 音量カーブの「対数（指数）化」
                # 線形(progress)ではなく、指数的(progress ** POWER_FACTOR)にすることで
                # 低速域で慎重に立ち上がり、高速域で豊かに広がる
                curve = progress ** POWER_FACTOR
                target_vol = curve * max_vol
                
            # 「底上げ」レイヤーの導入
            # 無音によるアンプのスリープ落ちを防ぐ ＆ 常に何かが流れている空気感を出す
            # レイヤー0は `BASE_RIPPLE_VOL` を下限とし、常に一定の音量が鳴り続ける
            if i == 0:
                target_vol = max(BASE_RIPPLE_VOL, target_vol)
                
            # 各レイヤーのボリューム変化もさらに小さくして滑らかに
            info = layers_info[i]
            info['target_vol'] = target_vol * GLOBAL_VOL
            info['current_vol'] += (info['target_vol'] - info['current_vol']) * VOL_CHANGE_SPEED
            
            # ボリューム反映
            info['channel'].set_volume(info['current_vol'])
                
        # --- 描画処理 ---
        screen.fill((30, 30, 40))
        
        # 波レイヤーの現在のボリュームを可視化（棒グラフ状）
        bar_width_unit = WINDOW_WIDTH // num_layers
        for i in range(num_layers):
            vol = layers_info[i]['current_vol']
            h = int((vol / GLOBAL_VOL) * (WINDOW_HEIGHT * 0.6))
            pygame.draw.rect(screen, (50 + i*20, 150, 255 - i*20), 
                             (i * bar_width_unit + 10, WINDOW_HEIGHT - h, bar_width_unit - 20, h))
                             
        text_speed = font.render(f"Walk Speed: {walk_speed:.2f}  Smooth: {smoothed_walk_speed:.2f}", True, (255, 255, 255))
        screen.blit(text_speed, (20, 20))
        
        mode_str = "UDP (Auto)" if operation_mode == 0 else "Mouse/Key (Manual)"
        text_state = font.render(f"Mode: {mode_str} | Parallel Layer Crossfade", True, (200, 200, 200))
        screen.blit(text_state, (20, 60))
        
        if operation_mode == 0:
            text_udp = font.render(f"UDP Raw: {udp_raw_data}", True, (150, 150, 200))
            screen.blit(text_udp, (20, 100))
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
