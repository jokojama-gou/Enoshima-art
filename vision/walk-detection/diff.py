import argparse
import cv2
import numpy as np
import time
import socket

def main():
    parser = argparse.ArgumentParser(description="Frame Difference - Walk Detection")
    parser.add_argument('--camera', type=int, default=None, help="Camera device ID. If not set, a selection menu will appear.")
    parser.add_argument('--udp-ip', type=str, default="127.0.0.1", help="UDP IP address to send step duration.")
    parser.add_argument('--udp-port', type=int, default=1902, help="UDP port to send step duration.")
    args = parser.parse_args()

    # UDPソケットの初期化
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

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

    def on_trackbar(val):
        pass

    window_name = "Diff Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Adjustable ParametersのTrackbarを作成
    # 差分のピクセル閾値 (この値以上の差があるピクセルを「動いた」とみなす)
    cv2.createTrackbar("Diff Threshold", window_name, 25, 255, on_trackbar)
    # 感度スケール (全体の何%が動いたら、出力値が1.0になるか。小さいほど高感度)
    # 例: 100なら、画面の10%が動くだけで値が1.0になる
    cv2.createTrackbar("Sensitivity (x100)", window_name, 200, 1000, on_trackbar)
    # FPS Limit
    cv2.createTrackbar("FPS Limit Delay", window_name, 30, 100, on_trackbar)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Unable to open camera {camera_id}")
        return

    # 前のフレームを保持する変数
    prev_gray = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # UIからのパラメータ取得
        diff_threshold = cv2.getTrackbarPos("Diff Threshold", window_name)
        sensitivity_val = cv2.getTrackbarPos("Sensitivity (x100)", window_name)
        sensitivity = sensitivity_val / 100.0 if sensitivity_val > 0 else 1.0
        fps_limit = cv2.getTrackbarPos("FPS Limit Delay", window_name)

        # グレースケール化とブラー（ノイズ除去）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # 最初のフレームの場合は初期化して次へ
        if prev_gray is None:
            prev_gray = gray
            continue

        # 前フレームとの差分を計算
        frame_diff = cv2.absdiff(prev_gray, gray)
        # 閾値で二値化（動いた部分が白(255)、それ以外が黒(0)になる）
        _, thresh = cv2.threshold(frame_diff, diff_threshold, 255, cv2.THRESH_BINARY)

        # 画面を左右に分割
        h, w = thresh.shape
        half_w = w // 2
        left_half = thresh[:, :half_w]
        right_half = thresh[:, half_w:]

        # 白ピクセル（動いたピクセル）の数をカウント
        left_motion_pixels = cv2.countNonZero(left_half)
        right_motion_pixels = cv2.countNonZero(right_half)

        # 全ピクセル数に対する割合 (0.0 ~ 1.0) を計算
        half_total_pixels = h * half_w
        left_ratio = left_motion_pixels / half_total_pixels
        right_ratio = right_motion_pixels / half_total_pixels

        # 感度を掛けて出力値とする (Sensitivityが10の場合、10%の動きで値が1.0になる)
        l_lift = min(5.0, left_ratio * sensitivity)
        r_lift = min(5.0, right_ratio * sensitivity)
        t_lift = l_lift + r_lift
        s_len = 0.0 # 歩幅は差分では計算できないため0.0固定

        # UDP送出
        try:
            msg = f"{l_lift:.4f},{r_lift:.4f},{t_lift:.4f},{s_len:.4f}".encode('utf-8')
            udp_sock.sendto(msg, (args.udp_ip, args.udp_port))
        except Exception as e:
            print(f"UDP Send Error: {e}")

        # 次のフレームのために保持
        prev_gray = gray.copy()

        # --- 描画処理 ---
        # 差分画像をRGBに変換して、元のフレームに重ねるか並べる
        # 今回は元のフレームに差分を赤色でオーバーレイ表示する
        thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        # 動いた部分を赤にする
        thresh_color[:, :, 0] = 0
        thresh_color[:, :, 1] = 0
        
        # 重ね合わせ
        output_frame = cv2.addWeighted(frame, 0.7, thresh_color, 0.3, 0)

        # パラメータとリアルタイム指標用の半透明オーバーレイ
        overlay = output_frame.copy()
        cv2.rectangle(overlay, (0, 0), (450, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, output_frame, 0.4, 0, output_frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        
        # 指標のテキスト描画
        cv2.putText(output_frame, f"L Motion: {l_lift:.3f}", (10, 30), font, 0.6, color, 1)
        cv2.putText(output_frame, f"R Motion: {r_lift:.3f}", (10, 80), font, 0.6, color, 1)
        cv2.putText(output_frame, f"Total (UDP t_lift): {t_lift:.3f}", (10, 130), font, 0.6, (200, 255, 200), 1)

        # リアルタイムの動き量を横向きバーで描画
        def draw_bar(y_pos, val, max_val=2.0):
            bar_max_w = 300
            bar_h = 15
            px = int(min(1.0, val / max_val) * bar_max_w)
            
            # バーの色
            bar_color = (0, 255, 255)
            
            cv2.rectangle(output_frame, (10, y_pos), (10 + px, y_pos + bar_h), bar_color, -1)
            cv2.rectangle(output_frame, (10, y_pos), (10 + bar_max_w, y_pos + bar_h), (255, 255, 255), 1)
            
        draw_bar(40, l_lift)
        draw_bar(90, r_lift)

        cv2.imshow(window_name, output_frame)
        
        # FPS上限調整と終了判定 (ESCキーで終了)
        delay = fps_limit if fps_limit > 0 else 1
        if cv2.waitKey(delay) & 0xFF == 27:
            break
                
    cap.release()
    cv2.destroyAllWindows()
    udp_sock.close()

if __name__ == '__main__':
    main()
