# ベンチマークの使い方

GPU の改善前後は [性能改善レポート](performance.md)、CPU を含む改善前の実測結果は [RTX 3090 × 2 の比較（2026-09-10）](benchmarks/rtx3090-2026-09-10.md) を参照してください。

すべての `benchmarks/bench_*.py` と `benchmark.py` が同じ CLI を使います。以下はリポジトリルートからの実行例です。コンテナ内では `docker compose exec -T tabai` を省いて実行できます。

## 単一 GPU・複数 GPU の指定

単一 GPU（GPU 0）で全演算を計測します。

```bash
docker compose exec -T tabai python benchmarks/benchmark.py --gpu-mode single --devices 0 --backends tabai --bits 1000000
```

同じ入力で GPU 1 枚と 2 枚を比較します。単一 GPU の列は `--devices` の先頭 GPU を使用します。

```bash
docker compose exec -T tabai python benchmarks/benchmark.py --gpu-mode both --devices 0 1 --backends tabai --bits 1000000 100000000 --pow-exponents 3 --output /mnt/benchmark-both.json
```

複数 GPU のみの乗算・二乗、または GPU 1 のみの累乗も計測できます。

```bash
docker compose exec -T tabai python benchmarks/bench_mul.py --gpu-mode multi --devices 0 1 --backends tabai --bits 100000000
docker compose exec -T tabai python benchmarks/bench_pow.py --gpu-mode single --devices 1 --backends tabai --bits 1000000 --pow-exponents 2 3 10
```

`single` は GPU ID を 1 個、`multi` / `both` は異なる GPU ID を `2, 4, ...` 個指定します。番号はプロセスから見える CUDA の番号です。`--devices` を省いた場合は単一モードで `[0]`、複数・比較モードで `[0, 1]` を使います。`--mode` は `--gpu-mode` の別名です。

加減算・比較は単一 GPU のままです。乗算・二乗・累乗・除算・剰余の内部乗算が分散対象になります。既定では長い方のオペランドが 64 Mbit 未満なら分散しません。小さい入力で分散経路そのものを確認する場合は `--min-bits 0` を指定します。ただし筆算方式の乗算は単一 GPU に残ります。

## スクリプトと演算

| スクリプト | 既定の演算 |
| --- | --- |
| `benchmark.py` | `add`, `sub`, `mul`, `square`, `div`, `mod`, `pow` |
| `bench_add.py` | 加算 |
| `bench_sub.py` | 同じビット数の正整数を `a > b` に揃えた減算 |
| `bench_mul.py` | 乗算と同一オブジェクトの二乗 |
| `bench_div.py` | 切り捨て除算と剰余 |
| `bench_pow.py` | 累乗 |
| `bench_multi_gpu.py` | 単一・複数 GPU の乗算・二乗・3 乗・除算の比較 |

`--operations mul square pow` のように演算を上書きできます。`div` は `//`、`mod` は `%` を表します。

`bench_multi_gpu.py` も共通処理を利用します。既定は `--gpu-mode both --backends tabai`、入力は 1 / 10 / 100 Mbit、指数は 3 です。1 枚しかない環境では `--gpu-mode single --devices 0` を指定します。

`python -m benchmarks.benchmark` や `python -m benchmarks.bench_mul` での起動にも対応しています。

## 計測の設定

| オプション | 既定値・意味 |
| --- | --- |
| `--backends` | `tabai`, `gmpy2`, `python` から選択。省略時はインポート可能なものを選び、複数 GPU の要求時は `tabai` を必須の計測対象に含める |
| `--bits` | 全選択演算の入力サイズを上書き。2 bit 以上、重複を除き昇順に計測 |
| `--pow-exponents` | 通常は `2 3 10 20`。非負整数を指定 |
| `--warmup` / `--repeat` | ウォームアップ 2 回、本計測 5 回。ウォームアップの時間は中央値に含めない |
| `--seed` | `42`。演算・入力サイズ・指数ごとに入力を決定 |
| `--timeout` | 各処理段階の上限 10 秒。正の有限値を指定 |
| `--min-bits` | 分散開始サイズ `64000000` |
| `--transfer` | `auto` または `host`。`auto` は可能なら P2P、それ以外は CPU 経由。`host` は CPU 経由を強制 |
| `--no-verify` | CPU の参照計算と結果照合を省略。既定では照合する |
| `--output` | JSON の保存先。親ディレクトリも作成する |

通常の既定サイズは、加減乗算・二乗が 1,000〜10,000,000,000 bit、除算・剰余が最大 1,000,000,000 bit、累乗の底が最大 10,000,000 bit です。最初は `--bits` で対象を絞ると確認しやすくなります。除算・剰余の除数は指定したサイズの半分です。

CPU だけを比較する場合は CuPy / CUDA を必要としません。CUDA が利用できない環境でも次のコマンドは実行できます。

```bash
docker compose exec -T tabai python benchmarks/benchmark.py --backends python gmpy2 --bits 1000 10000
```

`gmpy2` がない場合は `--backends python` を使います。明示したバックエンドの初期化に失敗した場合は、エラーとして記録します。

## 比較条件と出力

バックエンドごとに **独立した子プロセスを順番に起動**し、終了させてから次へ進みます。単一 GPU と複数 GPU の作業バッファが同時に GPU メモリを占有することを避けます。同じバックエンド内ではケース間でキャッシュを再利用します。

入力は seed・演算・ビット数・指数から生成するため、選択したバックエンドの順番や前のケースの成否に影響されません。各ケースの `input_sha256` が一致すれば同じ整数入力です。CPU 参照には GMP があれば GMP、なければ Python の `int` を使い、ウォームアップも含む各試行を照合します。

計測区間は演算呼び出しから GPU の完了同期までです。入力生成、CPU 参照計算、入力の GPU 変換、結果の CPU 変換・照合は含みません。分散に必要な GPU 間転送と同期は計測区間に含みます。CPU バックエンドの計測では CUDA を同期しません。前回の出力は次回の計測前に解放します。

進捗と失敗理由は標準エラー出力、最終比較表は標準出力に表示します。複数 GPU の結果には `NTT calls` を表示します。これはウォームアップ込みの分散乗算回数で、0 の行は分散による高速化を測った結果ではありません。

JSON は `schema_version: 2` です。`config` に CLI の設定、`reports` にバックエンド別の結果が入ります。各結果には次の情報を記録します。

- ケース：演算・サイズ・指数・入力ハッシュ・成否。
- 成功時：各計測の秒数、中央値、照合の有無、分散 NTT 回数、転送経路ごとの回数。
- GPU 情報：使用した GPU、CUDA / CuPy バージョン、P2P 可否、CuPy プールの予約バイト数。
- 失敗時：発生段階・例外の種類・理由。

GPU プールの予約量は前のケースのキャッシュも含み、CUDA コンテキストやページ固定ホストメモリを含む全体のピーク使用量ではありません。旧専用スクリプトによる [既存の測定ログ](benchmarks/multi-gpu-rtx3090.json) は当時の形式のまま保存しています。

## タイムアウトとエラー

`--timeout` は準備、CPU 参照計算、各ウォームアップ、各本計測、各結果照合、結果収集の段階ごとに適用します。上限を超えた子プロセスは停止します。初期化待ちの上限は `max(30, timeout)` 秒です。参照計算だけが上限に達する場合は、上限を延ばすか `--no-verify` を明示してください。

| 状態 | 意味 |
| --- | --- |
| `ok` | 計測成功 |
| `timeout` | 指定時間を超えてプロセスを停止 |
| `oom` | 入力変換・演算・参照計算などでメモリ不足 |
| `mismatch` | CPU 参照と結果が不一致 |
| `error` | 初期化失敗、非同期 CUDA エラーなどのその他の失敗 |
| `skipped` | 同じ演算・指数の小さい入力で timeout / OOM が起きたため省略 |

失敗したワーカーを再利用せず、必要な次のケースでは新しいプロセスを起動します。失敗結果も JSON に保存します。終了コードは成功が 0、計測失敗が 1、CLI 引数の不正が 2、Ctrl-C による中断が 130 です。Ctrl-C 時もワーカーを停止し、出力先を指定していれば途中までの結果を保存します。

この VM の CUDA 起動エラーへの環境指定は [複数 GPU 手順書](multi-gpu.md#この-vm-で確認した-cuda-起動エラー) を参照してください。
