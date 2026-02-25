int waitMilliSeconds = 1800;


void setup() {
  pinMode(13, OUTPUT);
  // シリアル通信の開始（ボーレートは9600）
  Serial.begin(9600);
}

void loop() {
  // シリアルバッファにデータがあるか確認
  if (Serial.available() > 0) {
    // データを読み捨ててバッファをクリア（Enterキー等の入力を検知）
    while(Serial.available() > 0) {
      Serial.read();
      delay(5); // 連続入力の取りこぼし防止
    }

    // トランジスタをON
    digitalWrite(13, HIGH);
    delay(waitMilliSeconds);

    // トランジスタをOFF
    digitalWrite(13, LOW);
    
    Serial.println("Action completed電流をながしました💡⏰️.");
    //enterキーおしたときにだけ、電流を通す。
  }
}