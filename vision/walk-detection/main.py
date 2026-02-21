# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mediapipe==0.10.14",
#     "opencv-python",
#     "numpy",
# ]
# ///

import argparse
import collections
import cv2
import mediapipe as mp
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Dict

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

@dataclass
class StepParams:
    height_threshold: float
    min_stable_frames: int
    dead_zone: float
    time_smoothing: int
    fps_limit: int

class PoseAnalyzer:
    def __init__(self, initial_params: StepParams):
        self.params = initial_params
        
        # 左右の足それぞれについて独立して状態を管理
        # 'IDLE': 接地（静止）状態
        # 'LIFTED': 足が上がった（離地）状態
        self.feet_state = {
            'left': {'state': 'IDLE', 'lift_time': 0.0, 'stable_frames': 0, 'history': collections.deque(maxlen=initial_params.time_smoothing)},
            'right': {'state': 'IDLE', 'lift_time': 0.0, 'stable_frames': 0, 'history': collections.deque(maxlen=initial_params.time_smoothing)}
        }
        
    def update_params(self, params: StepParams):
        self.params = params
        # Time Smoothing が変更された場合、移動平均の窓幅（dequeのmaxlen）を更新する
        for foot in self.feet_state.values():
            if self.params.time_smoothing > 0 and self.params.time_smoothing != foot['history'].maxlen:
                old_data = list(foot['history'])
                foot['history'] = collections.deque(old_data, maxlen=self.params.time_smoothing)
        
    def process_frame(self, results) -> tuple[Optional[float], Dict[str, float]]:
        metrics = {'left_lift': 0.0, 'right_lift': 0.0, 'step_length': 0.0}
        
        # 3Dワールド座標（実寸メートル）を使用することで遠近法の影響を排除
        if not results.pose_world_landmarks:
            return None, metrics
            
        landmarks = results.pose_world_landmarks.landmark
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        # y軸は下方向が正。両足のうち最も下に位置する方（yが大きい方）を「地面」とする
        ground_y = max(left_ankle.y, right_ankle.y)
        
        # 地面からの相対的な高さ（メートル）を算出
        raw_left = max(0.0, ground_y - left_ankle.y)
        raw_right = max(0.0, ground_y - right_ankle.y)
        
        # 歩幅（2D平面上の距離）の計算 (x, z軸: メートル)
        # ユーザーの「一歩進めたか」を補助的に確認するための指標
        step_length = np.sqrt((left_ankle.x - right_ankle.x)**2 + (left_ankle.z - right_ankle.z)**2)
        metrics['step_length'] = step_length
        
        self.feet_state['left']['history'].append(raw_left)
        self.feet_state['right']['history'].append(raw_right)
        
        completed_step_duration = None
        
        for foot_name in ['left', 'right']:
            foot = self.feet_state[foot_name]
            if len(foot['history']) == 0:
                continue
                
            # Time Smoothing: 直近の履歴から移動平均（ノイズ除去）を計算
            window = min(len(foot['history']), max(1, self.params.time_smoothing))
            recent_vals = list(foot['history'])[-window:]
            smoothed_lift = sum(recent_vals) / window
            metrics[f'{foot_name}_lift'] = smoothed_lift
            
            # State Machine: 状態遷移ロジック
            if foot['state'] == 'IDLE':
                # 離地 (Start): 静止状態からHeight Thresholdを超えて上昇した瞬間
                if smoothed_lift > self.params.height_threshold:
                    foot['state'] = 'LIFTED'
                    foot['lift_time'] = time.time()
                    foot['stable_frames'] = 0
            
            elif foot['state'] == 'LIFTED':
                # デバウンス処理: しきい値 - Dead Zone を下回った状態が連続するかチェック
                target_lower = max(0.0, self.params.height_threshold - self.params.dead_zone)
                
                if smoothed_lift < target_lower:
                    foot['stable_frames'] += 1
                else:
                    # 再び上昇した場合はカウンタをリセット
                    foot['stable_frames'] = 0
                    
                # 接地 (Complete): Min Stable Frames の間、連続して低い位置に留まった瞬間
                if foot['stable_frames'] >= self.params.min_stable_frames:
                    duration = time.time() - foot['lift_time']
                    # 複数人のうち「いずれかの個体の、いずれかの足」が動作完了すれば全体イベントとする
                    if completed_step_duration is None or duration > completed_step_duration:
                        completed_step_duration = duration
                    
                    # 状態を初期状態に戻す
                    foot['state'] = 'IDLE'
                    foot['stable_frames'] = 0

        return completed_step_duration, metrics

class Visualizer:
    def __init__(self):
        self.flash_frames_remaining = 0
        self.last_step_text = ""
        
    def trigger_flash(self, duration: float):
        self.flash_frames_remaining = 15 # 視覚的フィードバックの表示フレーム数
        self.last_step_text = f"[Step Detected] Duration: {duration:.2f}s"
        
    def draw(self, frame, results, metrics: Dict[str, float], params: StepParams):
        h, w, _ = frame.shape
        
        # 骨格とランドマークの描画
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
        # パラメータとリアルタイム指標用の半透明オーバーレイ
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (500, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        
        # 指標のテキスト描画
        cv2.putText(frame, f"L Lift(m): {metrics.get('left_lift', 0):.3f} (Thresh: {params.height_threshold:.3f})", (10, 30), font, 0.6, color, 1)
        cv2.putText(frame, f"R Lift(m): {metrics.get('right_lift', 0):.3f} (Thresh: {params.height_threshold:.3f})", (10, 80), font, 0.6, color, 1)
        cv2.putText(frame, f"Step Length(m): {metrics.get('step_length', 0):.3f}", (10, 140), font, 0.6, (200, 255, 200), 1)
        
        # リアルタイムの上がり幅を横向きバーで描画
        def draw_bar(y_pos, val, thresh, max_val=0.3):
            bar_max_w = 300
            bar_h = 15
            px = int(min(1.0, val / max_val) * bar_max_w)
            tx = int(min(1.0, thresh / max_val) * bar_max_w)
            
            # 閾値を超えていれば赤、超えていなければ緑
            bar_color = (0, 255, 0) if val < thresh else (0, 0, 255)
            
            cv2.rectangle(frame, (10, y_pos), (10 + px, y_pos + bar_h), bar_color, -1)
            cv2.rectangle(frame, (10, y_pos), (10 + bar_max_w, y_pos + bar_h), (255, 255, 255), 1)
            # 閾値のラインを描画
            cv2.line(frame, (10 + tx, y_pos - 5), (10 + tx, y_pos + bar_h + 5), (0, 255, 255), 2)
            
        draw_bar(40, metrics.get('left_lift', 0), params.height_threshold)
        draw_bar(90, metrics.get('right_lift', 0), params.height_threshold)
        
        # ステップ検知時の視覚的フィードバック (フラッシュ)
        if self.flash_frames_remaining > 0:
            text_size = cv2.getTextSize(self.last_step_text, font, 1.2, 2)[0]
            cx, cy = (w - text_size[0]) // 2, h // 2
            
            # 画面全体に枠線を描画
            cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 0), 10)
            
            # 中央に大きなテキストを描画（ドロップシャドウ付き）
            cv2.putText(frame, self.last_step_text, (cx + 2, cy + 2), font, 1.2, (0, 0, 0), 3)
            cv2.putText(frame, self.last_step_text, (cx, cy), font, 1.2, (0, 255, 0), 2)
            
            self.flash_frames_remaining -= 1
            
        return frame

def main():
    parser = argparse.ArgumentParser(description="MediaPipe Pose - Walk Detection")
    parser.add_argument('--camera', type=int, default=None, help="Camera device ID. If not set, a selection menu will appear.")
    parser.add_argument('--threshold', type=float, default=0.15, help="Initial height threshold for foot lift.")
    args = parser.parse_args()

    camera_id = args.camera
    if camera_id is None:
        import tkinter as tk
        from tkinter import ttk
        
        root = tk.Tk()
        root.title("Camera Setup")
        root.geometry("300x120")
        
        # 画面中央に配置
        root.eval('tk::PlaceWindow . center')
        
        selected_cam = tk.IntVar(value=0)
        
        tk.Label(root, text="利用するカメラの番号を選択してください:").pack(pady=10)
        
        # 0〜5番までのカメラを選択肢として用意（スキャンでのフリーズを避けるため固定長）
        display_values = [f"Camera {i}" for i in range(6)]
        combo = ttk.Combobox(root, values=display_values, state="readonly")
        combo.current(0)
        combo.pack(pady=5)
        
        def on_start():
            idx = combo.current()
            if idx >= 0:
                selected_cam.set(idx)
            root.destroy()
            
        tk.Button(root, text="Start", command=on_start, width=15).pack(pady=5)
        root.mainloop()
        
        camera_id = selected_cam.get()

    # Trackbarのイベントハンドラ（今回はTrackbarの値を取得するだけなのでpass）
    def on_trackbar(val):
        pass

    window_name = "Walk Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Adjustable ParametersのTrackbarを作成
    # 注: OpenCVのTrackbarは整数しか受け付けないため、小数は1000倍して扱う
    cv2.createTrackbar("Height Threshold (x1000)", window_name, int(args.threshold * 1000), 1000, on_trackbar)
    cv2.createTrackbar("Min Stable Frames", window_name, 5, 30, on_trackbar)
    cv2.createTrackbar("Dead Zone (x1000)", window_name, 20, 200, on_trackbar) # 初期値 0.02
    cv2.createTrackbar("Time Smoothing", window_name, 5, 30, on_trackbar)
    cv2.createTrackbar("FPS Limit Delay", window_name, 10, 100, on_trackbar)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Unable to open camera {camera_id}")
        return

    # 初期パラメータ
    params = StepParams(
        height_threshold=args.threshold,
        min_stable_frames=5,
        dead_zone=0.02,
        time_smoothing=5,
        fps_limit=10
    )
    
    analyzer = PoseAnalyzer(params)
    visualizer = Visualizer()
    
    # MediaPipe Pose の初期化
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Trackbarからの動的パラメータ更新
            try:
                params.height_threshold = cv2.getTrackbarPos("Height Threshold (x1000)", window_name) / 1000.0
                params.min_stable_frames = cv2.getTrackbarPos("Min Stable Frames", window_name)
                params.dead_zone = cv2.getTrackbarPos("Dead Zone (x1000)", window_name) / 1000.0
                smooth_val = cv2.getTrackbarPos("Time Smoothing", window_name)
                params.time_smoothing = smooth_val if smooth_val > 0 else 1
                params.fps_limit = cv2.getTrackbarPos("FPS Limit Delay", window_name)
                analyzer.update_params(params)
            except cv2.error:
                # ウィンドウが閉じられた場合などの例外処理
                pass

            # MediaPipeの処理のためにBGRをRGBに変換
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False 
            
            results = pose.process(image)
            
            # 描画等のために画像をBGRに戻す
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # ロジック判定と指標の取得
            step_duration, metrics = analyzer.process_frame(results)
            
            # ステップ判定時のコンソール出力と描画トリガー
            if step_duration is not None:
                print(f"[Step Detected] Duration: {step_duration:.2f}s")
                visualizer.trigger_flash(step_duration)

            # 描画処理
            output_frame = visualizer.draw(image, results, metrics, params)
            
            cv2.imshow(window_name, output_frame)
            
            # FPS上限調整と終了判定 (ESCキーで終了)
            delay = params.fps_limit if params.fps_limit > 0 else 1
            if cv2.waitKey(delay) & 0xFF == 27:
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
