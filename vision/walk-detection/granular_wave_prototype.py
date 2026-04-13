import pygame
import numpy as np
import random
import time
import os
import wave
import math
import sys

# --- Configuration ---
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 400
SAMPLE_RATE = 44100
CHANNELS = 2
NUM_LAYERS = 8      # 並列再生するレイヤー数
FADE_SEC = 0.5      # 各レイヤーのつなぎ目(ループ境界)のフェード処理
GLOBAL_VOL = 0.5    # マスターボリューム係数（音割れ防止）

SOURCE_FILE = 'source1.wav'

# --- Variables ---
walk_speed = 0.0
is_running = True
layers_info = []    # 各レイヤーのチャンネルや現在のボリュームなどの情報を保持

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
        
        # 永遠にループ再生開始（初期ボリュームは0なので聞こえない）
        channel.set_volume(0.0)
        channel.play(sound, loops=-1)
        
        layers_info.append({
            'channel': channel,
            'target_vol': 0.0,
            'current_vol': 0.0
        })
        print("Done")
    print("\n初期化完了。メインループを開始します。")

def main():
    global walk_speed, is_running
    
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
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT] or keys[pygame.K_UP]:
            walk_speed = min(1.0, walk_speed + 0.02)
        elif keys[pygame.K_LEFT] or keys[pygame.K_DOWN]:
            walk_speed = max(0.0, walk_speed - 0.02)
        else:
            if pygame.mouse.get_focused():
                mouse_x, _ = pygame.mouse.get_pos()
                walk_speed = np.clip(mouse_x / WINDOW_WIDTH, 0.0, 1.0)
                
        # --- 動的ボリューム制御ロジック ---
        num_layers = len(layers_info)
        
        for i in range(num_layers):
            # i階層目が鳴り始めるタイミングと最大になるタイミング
            # 例: 全8レイヤーでwalk_speedが上がりきるように調整
            activation_start = (i / num_layers) * 0.8
            activation_end = activation_start + 0.2
            
            if walk_speed <= 0.02:
                # 停止時は全階層を完全に0に向かわせる
                target_vol = 0.0
            elif walk_speed <= activation_start:
                target_vol = 0.0
            elif walk_speed >= activation_end:
                # そのレイヤーはMAX状態。全体がクリップしないようルートで除算
                target_vol = 1.0 / math.sqrt(i + 1)
            else:
                # フェードイン途中
                progress = (walk_speed - activation_start) / (activation_end - activation_start)
                target_vol = progress * (1.0 / math.sqrt(i + 1))
                
            # 急激な変化を防ぐ慣性（LERP）
            info = layers_info[i]
            info['target_vol'] = target_vol * GLOBAL_VOL
            info['current_vol'] += (info['target_vol'] - info['current_vol']) * 0.1
            
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
                             
        text_speed = font.render(f"Walk Speed: {walk_speed:.2f}", True, (255, 255, 255))
        screen.blit(text_speed, (20, 20))
        
        text_state = font.render(f"Simulation: Parallel Layer Crossfade", True, (200, 200, 200))
        screen.blit(text_state, (20, 60))
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
