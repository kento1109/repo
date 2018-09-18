GAN(Recurrent GAN)
---
- RGAN_sin：RGANでsin波を生成
- RGAN_mnist：RGANでMNIST画像を生成
- RCGAN_sin：RCGANでsin波を生成（ラベルごとに異なる周波数で波を作り、そのラベルを条件に与える）
- RCGAN_mnist：RCGANでMNIST画像を生成（画像のラベルを条件に与える）
- TSTR：TRAIN ON SYNTHETIC, TEST ON REAL（生成したデータの評価）

### 評価について
TSTRでは、訓練用に合成データ、評価用に正解データを利用する。
- synthetic_data.npz：RCGANにて生成したデータ
- true_data.npz：RCGANの学習元として使用した正解データ
