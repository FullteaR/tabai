# 大きな整数演算の性能改善（2026-09-10）

NTT の全配列読み書きと CPU 経由の GPU 間転送を改善した。1 億 bit の単一 GPU 乗算は 36.360 → 19.277 ms（1.89 倍）、2 GPU 乗算は 32.661 → 19.017 ms（1.72 倍）になった。

## ボトルネックと修正

CUDA イベントで各 GPU の NTT・桁上がり・結果の整形を測り、転送は別途同期完了までの時間を測定した。プロファイル用のフックは通常のベンチマークには入れていない。

| 対象 | 改善前の実測 | 修正 | 改善後の実測 |
| --- | --- | --- | --- |
| 単一 GPU の NTT（1 億 bit の乗算） | 順変換 2 回＋逆変換 1 回で 32.31 ms。演算全体 36.33 ms の約 89% | 大域的な 2 段をレジスタで融合。末尾の順変換／先頭の逆変換は最大 1,024 係数を共有メモリで処理 | NTT 合計 15.94 ms |
| CPU 経由の 64 MiB 転送（GPU 1 → 0） | 全体を読み出してから全体を書き込み、10.34 ms | 2 MiB ごとに分け、次の D2H と前の H2D を重ねる | 5.63 ms |
| 単一 GPU の桁上がり（1 億 bit の乗算） | NTT のゼロ埋めを含む 16,777,217 チャンクを処理、2.31 ms | 線形畳み込みと桁上がり用の 12,500,000 チャンクだけを処理 | 1.74 ms |

変換長が 2^24 の NTT は、24 回の全配列処理を 8 回のカーネル起動にまとめた。各バタフライの Goldilocks 素数上の演算と出力順序は同じで、浮動小数点近似は使用していない。

CPU 転送バッファはページ固定メモリを再利用する。送信元の計算完了を確認してから専用ストリームで分割読み出しを行い、各チャンクの完了後に受信 GPU へ書き込む。アップロードの完了イベントを記録し、バッファを再利用・拡張する直前に待つ。後続の NTT 計算まで待つ必要がなくなった。例外時とコンテキスト終了時もストリームを完了させてから所有バッファを解放する。

分割サイズは 1/2/4/8/16 MiB と分割なし、共有メモリの処理単位は 128/256/512/1,024 係数で比較した。この RTX 3090 構成の大規模処理で速かった 2 MiB と 1,024 係数を採用した。

## 通常ベンチマークの改善前後

同じ VM の RTX 3090 24 GiB × 2、PHB 接続、P2P 不可。CuPy 14.0.1 / CUDA 13.0 / driver 580.173.02。CPU 参照は gmpy2 2.3.1（GMP 6.3.0）。各ケースを別バックエンドプロセスで順番に測定し、ウォームアップ 2 回、本計測 7 回の中央値を比較した。

入力は 1 億 bit。除算の除数は 5,000 万 bit、累乗の指数は 3。時間は ms、倍率はそれぞれのモードの改善前 ÷ 改善後。

| 演算 | 単一 GPU 前 | 単一 GPU 後 | 改善倍率 | 2 GPU 前 | 2 GPU 後 | 改善倍率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 乗算 | 36.360 | 19.277 | 1.89× | 32.661 | 19.017 | 1.72× |
| 二乗 | 25.256 | 13.589 | 1.86× | 26.218 | 14.944 | 1.75× |
| 3 乗 | 104.851 | 55.979 | 1.87× | 89.785 | 51.643 | 1.74× |
| 除算 | 239.075 | 125.450 | 1.91× | 230.594 | 128.173 | 1.80× |

入力の GPU 変換・CPU 参照計算・結果回収と照合は時間に含めない。演算中の GPU 間転送と完了同期は含む。初回の CUDA 初期化・カーネルコンパイル・テーブル生成を含む時間の比較ではない。

100 万 bit も同じ設定で計測し、生データに保存した。小さい入力は VM のスケジューリングによるばらつきの影響が大きいため、上の改善倍率は大規模入力について示している。

## さらに大きな入力

5.12 億 bit は、同日先に保存した改善前の測定ログと比較した。設定と入力ハッシュは一致し、その測定時のライブラリ全ファイルの SHA-256 も今回の改善前コードと一致することを確認した。

| 演算 | モード | 改善前（ms） | 改善後（ms） | 改善倍率 |
| --- | --- | ---: | ---: | ---: |
| 乗算 | 単一 GPU | 176.008 | 99.240 | 1.77× |
| 二乗 | 単一 GPU | 121.230 | 69.852 | 1.74× |
| 乗算 | 2 GPU | 146.554 | 89.870 | 1.63× |
| 二乗 | 2 GPU | 115.306 | 68.512 | 1.68× |

## 残るボトルネックと使い分け

単一 GPU 側も大きく速くなったため、GPU を 2 枚使う追加効果は小さくなった。この VM では転送と先頭 GPU の桁上がり処理が残る。1 億 bit の乗算は単一・2 GPU がほぼ同等で、二乗・除算は単一 GPU の方が速い測定結果だった。5.12 億 bit の乗算は 99.240 → 89.870 ms で、2 GPU が約 1.10 倍速い。

既定の分散開始サイズ 64 Mbit は当初の実装の測定から選んだ値で、今回の実装では 6,400 万 bit の分散乗算・二乗は単一 GPU より遅い。公開 API の既定値は維持し、入力サイズと演算に応じて `min_bits` / `--min-bits` を指定できる。常に 2 GPU が有利になる閾値とは解釈しない。物理 P2P・実機 4 GPU での性能は未測定。

## 検証

- `pytest tests/ --require-multi-gpu -q`：**648 passed, 2 skipped**（68.11 秒）。スキップは重複 GPU ID の公開 API テストで、実機 2 GPU の検証は実行済み。
- 順変換の参照比較を融合段階の境界まで拡張。逆変換は独立した Python 参照と比較し、配列の前後を上書きしないことも確認。
- CPU 転送バッファを異なるストリームで続けて使い、サイズ拡張・端数チャンク・再利用後も結果が壊れないことを実機で確認。
- 今回の通常ベンチマーク計 48 ケース、336 回の本計測と 96 回のウォームアップはすべて GMP と一致。入力ハッシュも改善前後で一致し、失敗・タイムアウト・OOM なし。

## 再実行とログ

リポジトリルートから実行する。

```bash
docker compose exec -T -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64 tabai python benchmarks/benchmark.py --gpu-mode both --devices 0 1 --backends tabai --operations mul square pow div --bits 1000000 100000000 --pow-exponents 3 --warmup 2 --repeat 7 --timeout 30 --output /mnt/performance.json

docker compose exec -T -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64 tabai python benchmarks/bench_mul.py --gpu-mode both --devices 0 1 --backends tabai --bits 64000000 100000000 256000000 512000000 --warmup 2 --repeat 7 --timeout 30 --output /mnt/performance-scaling.json
```

- [改善前](benchmarks/optimization-2026-09-10/performance-before.json) / [改善後](benchmarks/optimization-2026-09-10/performance-after.json) / [改善後のサイズ比較](benchmarks/optimization-2026-09-10/performance-scaling-after.json)。
- [改善前の大規模測定](benchmarks/rtx3090-2026-09-10-scaling.json)。
- [CUDA イベント計測：前](benchmarks/optimization-2026-09-10/profile-before.json) / [後](benchmarks/optimization-2026-09-10/profile-after.json)。異なる GPU のイベント時間は重なり得るため、合算して全体時間とは扱わない。
- [処理単位・転送サイズの比較](benchmarks/optimization-2026-09-10/performance-tuning.json) / [環境・ソースの識別情報](benchmarks/optimization-2026-09-10/provenance.json)。
- [調査用プロファイラ](benchmarks/optimization-2026-09-10/profile_hotpath.py) / [比較用スクリプト](benchmarks/optimization-2026-09-10/tune_hotpath.py)。通常の CLI と異なり、GPU 内部処理を調べる開発用スクリプト。

調査用プロファイラは次のようにコンテナへ渡せる。計測用フックとキャッシュを使うため、性能比較には上の通常ベンチマークを使用する。

```bash
docker compose exec -T -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64 tabai python - --output /mnt/profile.json < docs/benchmarks/optimization-2026-09-10/profile_hotpath.py
```
