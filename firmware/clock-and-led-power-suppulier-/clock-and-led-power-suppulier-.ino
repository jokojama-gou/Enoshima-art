void setup() {
  // 13番ピンを出力モードに設定
  pinMode(13, OUTPUT);
}

void loop() {
  // トランジスタをONにする（門を開く）
  digitalWrite(13, HIGH);
  delay(1000); // 1秒待機

  // トランジスタをOFFにする（門を閉じる）
  digitalWrite(13, LOW);
  delay(1000); // 1秒待機
}