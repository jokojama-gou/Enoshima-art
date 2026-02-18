# Enoshima-art




2. システム構成 (System Architecture)
本システムは、機能ごとに分離された3つの層（Layer）で構成される。

A. Vision Layer (メインPC)
役割: 映像解析および波の映像生成。

技術: Python (OpenCV/MediaPipe) または TouchDesigner。

機能: センサー（カメラ/ミリ波レーダー）から人物の移動速度を算出し、OSCメッセージとしてGatewayへ送信。同時に、その速度に応じた波の映像・音響を生成・出力する。出力はプロジェクターから行われる。

B. Gateway Layer (Raspberry Pi)
役割: システム全体のステート管理と翻訳。

技術: Python。

機能: Vision Layerからの歩行データを受け取り、作品全体の「時間進行度」を算出。その値をActuator Layerが解釈可能な形式に変換し、シリアル通信等で伝達する。

C. Actuator Layer (MCU / マイコンボード)
役割: 物理デバイスの駆動。

技術: C++ (Arduino / ESP32)。

機能: 受け取った進行データに基づき、ステッピングモータを制御して掛け時計の針を物理的に進める。

3. ディレクトリ構成 (Repository Structure)


├── docs/                # 企画書、回路図、通信プロトコル定義(OSC/Serial)
├── vision/              # 映像処理・人物検知プログラム
│   ├── tracking/        # センサー解析用ソースコード
│   └── rendering/       # 映像生成（波のシミュレーション）ファイル
├── gateway/             # Raspberry Pi 司令塔プログラム
│   └── src/             # ロジック・ステート管理
├── firmware/            # マイコン用ソースコード（時計制御用）
├── monitoring/          # 遠隔監視・生存確認用スクリプト
└── tools/               # 設営・キャリブレーション用補助ツール


4. 通信仕様 (Communication Protocol)
各モジュール間は以下のプロトコルで越境的に接続される。

Vision -> Gateway: OSC (/messidor/velocity, float)

Gateway -> Actuator: Serial (Custom Packet: [STEP_COUNT]\n)

5. 構築・展開方法 (Deployment)
各ディレクトリ内の README.md を参照のこと。依存関係の解決には、不確実性を排除するため以下の管理ツールを墨守すること。

Python: requirements.txt

Node.js: package.json

6. 著者 (Author)
横山 豪 (Go Yokoyama)

慶應義塾大学 総合政策学部