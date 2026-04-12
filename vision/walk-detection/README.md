# MediaPipe Pose - Walk Detection (歩行検知システム)

This project uses MediaPipe Pose to detect human walking steps from a camera feed in real-time. It provides visual feedback and adjustable parameters fine-tuned for stable step detection in exhibition environments.

このプロジェクトは、MediaPipe Poseを使用してカメラ映像から人間の歩行動作をリアルタイムに検知するシステムです。展示環境などでの安定したステップ検知のために微調整可能なパラメータと視覚的フィードバックを提供します。

## Requirements (必須要件)
- Python >= 3.11
- mediapipe == 0.10.14
- opencv-python
- numpy

## Installation (インストール)

This project supports [uv](https://github.com/astral-sh/uv), an extremely fast Python package installer and resolver. The dependencies are defined as inline script metadata (`# /// script`) within `main.py` and managed via `uv.lock`. You can run the program immediately without manual installation if you use `uv run`.

このプロジェクトは高速なPythonパッケージ管理ツールである [uv](https://github.com/astral-sh/uv) をサポートしています。依存関係は `main.py` 内のインラインメタデータ（`# /// script`）で定義され、`uv.lock` で固定・管理されています。そのため、`uv run` コマンドを使用すれば、事前のライブラリインストール作業なしで即座に実行可能です。

### Using uv (Recommended / 推奨)
```bash
# Just run it; uv downloads specific python version and dependencies automatically
# 実行するだけで、uvが必要なPythonバージョンと依存パッケージを自動で用意してくれます
uv run main.py
```

### Using pip (Standard / 通常)
```bash
pip install mediapipe==0.10.14 opencv-python numpy
```

## Usage (起動方法)

To run the script:
スクリプトを実行するには以下のコマンドを実行します:

### With uv (推奨):
```bash
uv run main.py
```

### With standard Python (通常):
```bash
python main.py
```

### Command-line Arguments (コマンドライン引数)

You can specify the camera ID and initial height threshold directly:
カメラIDと初期の高さしきい値を直接指定して起動することも可能です:

```bash
python main.py --camera 0 --threshold 0.15
```

- `--camera`: Camera device ID (e.g., `0`, `1`). If not provided, a GUI selection menu will appear.
  (カメラデバイスID。指定がない場合はカメラ選択用のGUIメニューが表示されます。)
- `--threshold`: Initial height threshold for foot lift (in meters). Default is `0.15`.
  (足が離地したと判定する初期の高さしきい値(メートル)。デフォルトは `0.15` です。)

### Controls (操作方法)

- Press **ESC** key in the camera window to exit the application.
  (カメラウィンドウ上で **ESC** キーを押すとアプリケーションを終了します。)

## Adjustable Parameters (リアルタイム調整パラメータ)

The application opens a window with OpenCV trackbars to adjust detection logic dynamically:
アプリケーション実行時、OpenCVのウィンドウ上にトラックバーが表示され、実行したまま検知ロジックのパラメータを動的に調整できます:

- **Height Threshold (x1000)**: The physical height (in mm) the foot needs to rise to be considered "lifted".
  (足が「浮いた（離地）」状態へ遷移するために必要な高さのしきい値(ミリ換算)。)
- **Min Stable Frames**: The number of consecutive frames the foot must stay below the threshold to be considered "landed".
  (足がしきい値を下回った状態が連続して何フレーム続けば「着地完了」と判定するかの値。)
- **Dead Zone (x1000)**: Buffer zone (in mm) below the threshold to prevent chattering. Land condition is `threshold - dead_zone`.
  (チャタリング現象を防ぐための緩衝帯(ミリ換算)。着地判定ラインは `全体しきい値 - Dead Zone` になります。)
- **Time Smoothing**: The number of recent frames used to calculate the moving average of the foot's height, reducing noise.
  (骨格推定のブレ（ノイズ）を減らすために、足の高さの移動平均を計算する直近のフレーム数。)
- **FPS Limit Delay**: Artificial delay in milliseconds added to the frame processing loop to control CPU usage and framerate.
  (処理ループに挿入される待機時間(ミリ秒)。CPUの負荷軽減と実効FPSの制御に使われます。)

> **Note**: For in-depth details on parameter tuning strategies and their impacts, please refer to [Manual-adjustment.md](./Manual-adjustment.md).
> **注意**: 各パラメータの具体的な調整方針や環境ごとの対策については、[Manual-adjustment.md](./Manual-adjustment.md) の詳細解説を参照してください。

## Architecture (内部構造)

- `PoseAnalyzer`: Handles the state-machine logic for detecting steps based on 3D world landmarks (which mitigates perspective distortion).
  (3D座標を使用して遠近法の影響を排除しつつ、足の離地・接地の状態遷移（ステートマシン）を管理・判定します。)
- `Visualizer`: Renders the skeletons, real-time height indicators, and full-screen visual flash feedback upon confirmed steps.
  (骨格の描画、リアルタイムの上がり幅を示すバー、さらにステップ検知時の画面全体へのフラッシュフィードバック等の描画を行います。)
