# tabai

`tabai` は、**大きな整数の演算を NVIDIA GPU で高速化することを目指す Python の多倍長整数ライブラリ**です。CuPy 経由で独自の CUDA カーネルを実行し、符号付き整数を `TabaiInt` クラスで扱います。Python 標準の `int` と `gmpy2.mpz`（GMP）との性能比較用ベンチマークも含まれています。

整数を GPU 上に保持して演算を繰り返し、必要なときに Python の `int` に戻す構成です。GPU の起動・同期・転送コストがあるため、すべての入力サイズで CPU より速いことを保証するものではありません。

## 対応する操作

| 操作 | API |
| --- | --- |
| Python 整数から生成／Python 整数へ変換 | `TabaiInt(value)` / `value.to_cpu()` |
| 加算・減算・乗算 | `+`, `-`, `*` |
| 切り捨て除算・剰余 | `//`, `%`, `divmod(a, b)` |
| 非負整数乗 | `**`, `pow(a, b)` |
| 符号反転・絶対値 | `-a`, `abs(a)` |
| 比較 | `==`, `!=`, `<`, `<=`, `>`, `>=` |

演算・比較では Python の `int` と混在できます。負数の除算は Python と同じく負の無限大方向へ丸め、ゼロでない余りの符号は除数に従います。ゼロ除算は `ZeroDivisionError`、負の指数は `ValueError` になります。

`/`、ビット演算・シフト、3 引数の `pow(a, b, modulus)`、`int(a)` による変換は公開クラスに実装されていません。Python 整数への変換には `to_cpu()` を使います。

## 実行環境と使用例

ホストには NVIDIA GPU とドライバー、Docker Compose、および Docker から GPU を利用できる設定（NVIDIA Container Toolkit など）が必要です。

`Dockerfile` は `cupy/cupy:v14.0.1` を使用し、仮想環境に `pytest` と `gmpy2` を追加します。`./tabai` はコンテナの `/mnt` にマウントされ、`/mnt/src` が `PYTHONPATH` に追加されます。パッケージインストール用の `pyproject.toml` / `setup.py` はなく、現在はソースを直接読み込む構成です。

リポジトリのルートで実行します。

```bash
docker compose up -d --build
docker compose exec tabai bash
```

以下の Python 例はコンテナ内で実行できます。

```python
from tabai_gpu import TabaiInt

a = TabaiInt((1 << 4096) - 1)
b = TabaiInt((1 << 2048) + 1)
product = a * b
assert product.to_cpu() == ((1 << 4096) - 1) * ((1 << 2048) + 1)

q, r = divmod(TabaiInt(-7), 3)
assert (q.to_cpu(), r.to_cpu()) == (-3, 2)
assert (2 ** TabaiInt(100)).to_cpu() == 1 << 100
assert (a + 1) > a
```

演算時にデバイス・ストリーム別の GPU エンジンを遅延生成するため、動作する CuPy / CUDA 環境が必要です。ライブラリ全体を CPU のみで動かすフォールバックはありません。

## 複数 GPU で 1 回の演算を処理する

大きな乗算の NTT を複数 GPU に分割できます。累乗・除算の内部乗算にも適用されます。

```python
from tabai_gpu import TabaiInt, multi_gpu

a = TabaiInt((1 << 2_000_000) - 1)  # 現在の GPU が 0 の場合
b = TabaiInt((1 << 1_000_000) + 7)
with multi_gpu([0, 1], min_bits=0):  # 小さめの例でも分散を有効にする
    product = a * b
```

既定では 64 Mbit 未満の乗算を単一 GPU に残します。加減算や最終的な桁上がり処理は先頭の GPU で行います。**RTX 3090 × 2 の実機で、CPU 経由の転送と演算の正確性を検証済みです。** 性能改善後の 512 Mbit の乗算は、単一 GPU 99.2 ms、2 GPU 89.9 ms（約 1.10 倍）でした。改善前後の計測と残るボトルネックは [性能改善レポート](docs/performance.md) を参照してください。高速化率は演算と入力サイズに依存し、P2P 転送はこの VM では利用できないため未検証です。 設定、制約、載せ替え後の検証コマンドは [複数 GPU の使用・検証手順](docs/multi-gpu.md) を参照してください。

## 実装の構成

| ファイル | 役割 |
| --- | --- |
| [tabai_int.py](tabai/src/tabai_gpu/tabai_int.py) | 公開クラス、符号管理、演算子、Python `int` との相互運用 |
| [core.py](tabai/src/tabai_gpu/core.py) | `GPUBigInt` と CUDA カーネル、作業バッファ・NTT テーブルの再利用 |
| [multi_gpu.py](tabai/src/tabai_gpu/multi_gpu.py) | NTT の GPU 間分割・転送と `multi_gpu` コンテキスト |
| [utils.py](tabai/src/tabai_gpu/utils.py) | Python 整数と GPU 配列の相互変換 |
| [tests/](tabai/tests/) | 演算結果、符号、境界条件、NTT の検証 |
| [benchmarks/](tabai/benchmarks/) | 演算別の性能比較と共通の計測処理 |

絶対値は下位桁から並ぶ `uint32` 配列として GPU に保持し、符号は Python 側で別に管理します。

- **加減算**: 並列プレフィックススキャンで桁上がり・桁借りを伝播します。
- **乗算**: 両オペランドの 32 bit 桁数の積が `6016 * 6016` 以下なら筆算方式、それを超えると Goldilocks 素数 `p = 2^64 - 2^32 + 1` 上の NTT（数論変換）を使います。どちらも 16 bit チャンクの列和を求め、共通の桁上がり処理で整数に戻します。NTT は DIF 順変換 / DIT 逆変換を使い、大域的な 2 段の融合と最大 1,024 係数の共有メモリ処理で全配列の読み書きを減らします。同一配列の二乗では順変換を 1 回に省略します。
- **除算・剰余**: 逆数から商の候補を求め、余りを使って補正します。大きな精度では GPU の Newton 反復を使いますが、小さな精度や条件に応じて逆数を CPU の整数除算で求めます。
- **累乗**: 指数を CPU で読み取り、スライディングウィンドウ法で GPU の乗算・二乗を繰り返します。

NTT 乗算は浮動小数点 FFT の丸めに依存しません。ただし数学上の変換長は `2^32` 以下で、列和が `int64` に収まる条件も必要です。実装は `min(16 bit チャンク数) * 65535**2 < 2**63` を assert していますが、単一 GPU 経路では変換長の上限を明示的に検査していません。分散経路は変換長・列和・CUDA 添字の上限を検査します。実際の扱えるサイズは GPU メモリにも制約されます。共有エンジンは作業バッファと変換長ごとのテーブルを保持し、作業バッファは拡張後に自動縮小しません。

## テスト

ホスト側のリポジトリルートから実行します。

```bash
docker compose exec tabai pytest tests/ -q
docker compose exec tabai pytest tests/test_tabai_int.py -q
docker compose exec tabai pytest tests/test_ntt.py -q
```

Python の整数演算との照合に加え、桁上がり、演算方式の切り替え境界、作業バッファの再利用、NTT の法演算・往復変換・参照実装との一致を検証します。大きな整数を扱うテストも含むため、実行時間と必要メモリは環境に依存します。

## ベンチマーク

`--backends` で測定対象を指定します。CPU と GPU に同じ整数入力を与え、演算ごとの時間を比較できます。

| 指定値 | 測定対象 |
| --- | --- |
| `python` | CPU 上の Python 標準 `int` |
| `gmpy2` | CPU 上の `gmpy2.mpz`（GMP）。Docker イメージにインストール済み |
| `tabai` | GPU 上の `TabaiInt`。`--gpu-mode single\|multi\|both` で単一・複数・両者の比較を選択 |

以下はホスト側のリポジトリルートから実行します。

```bash
# CPU の Python int と GMP で全演算を比較
docker compose exec -T tabai python benchmarks/benchmark.py --backends python gmpy2 --bits 1000 10000 --pow-exponents 3 --output /mnt/benchmark-cpu.json

# 単一 GPU（GPU 0）・GMP・Python int の 3 者を比較
docker compose exec -T tabai python benchmarks/benchmark.py --gpu-mode single --devices 0 --backends tabai gmpy2 python --bits 1000000 --pow-exponents 3 --warmup 2 --repeat 7 --output /mnt/benchmark-single-cpu.json

# 単一 GPU・2 GPU・GMP・Python int の 4 者を比較
docker compose exec -T tabai python benchmarks/benchmark.py --gpu-mode both --devices 0 1 --backends tabai gmpy2 python --bits 1000000 --pow-exponents 3 --warmup 2 --repeat 7 --output /mnt/benchmark-both-cpu.json

# 1 億 bit の乗算・二乗を単一 GPU・2 GPU・GMP で比較
docker compose exec -T tabai python benchmarks/bench_mul.py --gpu-mode both --devices 0 1 --backends tabai gmpy2 --bits 100000000 --warmup 2 --repeat 7 --timeout 30 --output /mnt/benchmark-large.json
```

`benchmark.py` は加算・減算・乗算・二乗・除算・剰余・累乗を計測します。上の例では累乗の指数を `--pow-exponents 3` で 3 に絞っています。`bench_mul.py` は乗算・二乗のみです。`bench_add.py`、`bench_sub.py`、`bench_div.py`、`bench_pow.py`、`bench_multi_gpu.py` も同じ CLI を使い、`--backends python gmpy2` などを指定できます。

`both` の比較表には `tabai-single`、`tabai-multi`、`gmpy2`、`python` の列が並びます。単一 GPU は `--devices` の先頭 GPU を使用します。既定では 64 Mbit 未満の乗算は分散しないため、100 万 bit の例は複数 GPU モードでも GPU 0 で演算します。小さい入力で分散経路を測る場合は `--min-bits 0` を追加し、出力の `NTT calls` が 0 より大きいことを確認してください。

大きな入力では Python `int` の計算に時間がかかるため、最後の例は CPU 比較対象を GMP に絞っています。Python も含める場合は `--backends tabai gmpy2 python` に変更し、必要に応じて `--timeout` を延ばしてください。時間制限は準備・各演算・各照合などの段階ごとに適用され、タイムアウトではワーカーを停止します。同じ演算・指数のより大きい入力はスキップされます。

CPU バックエンドの計測には CuPy / CUDA は不要です。Docker を使わず、ローカルの Python で次のようにも実行できます。GMP を含める場合はその Python 環境に `gmpy2` が必要です。

```bash
python tabai/benchmarks/benchmark.py --backends python gmpy2 --bits 1000 10000 --pow-exponents 3
# gmpy2 がない場合は Python int のみ
python tabai/benchmarks/benchmark.py --backends python --bits 1000 10000 --pow-exponents 3
```

この VM で GPU の初期化時に `cudaErrorCompatNotSupportedOnDevice` が出る場合は、GPU を使うコマンドの `docker compose exec -T tabai` を次の環境指定付きに置き換えます。

```bash
docker compose exec -T -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64 tabai python benchmarks/bench_mul.py --gpu-mode both --devices 0 1 --backends tabai gmpy2 python --bits 1000000
```

バックエンドごとに別プロセスで測定し、ウォームアップを除く本計測の中央値を表示します。GPU は演算完了までを測り、分散に必要な転送と同期も含みます。入力の変換と CPU 参照による結果照合は計測区間外です。既定で各試行の結果を照合し、メモリ不足・計算不一致・その他のエラーも記録します。

`--output /mnt/benchmark-cpu.json` の結果は、ホストでは `tabai/benchmark-cpu.json` に保存されます。JSON には各試行時間・中央値・入力ハッシュ・照合結果などが入ります。

全オプション・出力形式・時間制限の詳細は [ベンチマークの使い方](docs/benchmarks.md)、改善後の実測値は [性能改善レポート](docs/performance.md)、CPU を含む改善前の比較は [RTX 3090 × 2 と CPU の測定結果](docs/benchmarks/rtx3090-2026-09-10.md) を参照してください。

## 関連ドキュメント

- [CLAUDE.md](CLAUDE.md): 開発時の構成・実装上の注意点。
- [NTT 移行手順書](docs/ntt-migration.md): FFT から NTT への移行時の設計・検証計画。現在の乗算実装は NTT に切り替わっています。

## ライセンス

[MIT License](LICENSE)
