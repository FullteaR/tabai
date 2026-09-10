# tabai

**GPU で動く Python の多倍長整数ライブラリ。**
`tabai` は CuPy 経由で独自の CUDA カーネルを実行し、大きな整数を GPU 上に保持したまま演算します。
Python の `int` と同じ感覚で使える `TabaiInt` クラスを提供し、巨大な乗算は NTT（数論変換）で計算して、
複数の GPU に分散することもできます。

[English README](README.md)

## 対応する操作

| 操作 | API |
| --- | --- |
| Python 整数から生成／Python 整数へ変換 | `TabaiInt(value)` / `value.to_cpu()` |
| 加算・減算・乗算 | `+`, `-`, `*` |
| 切り捨て除算・剰余 | `//`, `%`, `divmod(a, b)` |
| 非負整数乗 | `**`, `pow(a, b)` |
| 符号反転・絶対値 | `-a`, `abs(a)` |
| 比較 | `==`, `!=`, `<`, `<=`, `>`, `>=` |

いずれも Python の `int` と混在して使えます。除算は Python と同じく負の無限大方向へ丸め、
ゼロでない余りの符号は除数に従います。

## インストール

必要な環境は、NVIDIA GPU とドライバー、Docker Compose、Docker から GPU を使うための設定
（NVIDIA Container Toolkit など）です。

```bash
git clone https://github.com/FullteaR/tabai.git
cd tabai
docker compose up -d --build
docker compose exec tabai bash
```

コンテナは `cupy/cupy:v14.0.1` をベースに、`./tabai` を `/mnt` にマウントし、`/mnt/src` を `PYTHONPATH`
に追加します。そのまま `import tabai_gpu` できます。pip パッケージはまだありません。

GPU の初期化で `cudaErrorCompatNotSupportedOnDevice` が出る場合は、ホスト側の CUDA ライブラリを
コンテナの環境変数に追加します。

```bash
docker compose exec -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64 tabai bash
```

## 使い方

```python
from tabai_gpu import TabaiInt

a = TabaiInt((1 << 4096) - 1)
b = TabaiInt((1 << 2048) + 1)

product = a * b
print(product.to_cpu())          # Python の int に戻す

q, r = divmod(TabaiInt(-7), 3)   # (-3, 2) — Python と同じ結果
big = a ** 3
mixed = a + 1                    # Python の int をそのまま渡せる
assert mixed > a
```

### 複数 GPU を使う

`multi_gpu` のスコープ内では、大きな乗算の NTT が指定した GPU に分割されます
（`**`、`//`、`%` の内部で行われる乗算にも適用されます）。

```python
from tabai_gpu import TabaiInt, multi_gpu

a = TabaiInt((1 << 200_000_000) - 1)
b = TabaiInt((1 << 200_000_000) + 7)

with multi_gpu([0, 1]):
    product = a * b
```

既定では 64 Mbit 未満の乗算は単一 GPU で処理します。小さい入力も分散させる場合は `min_bits=0` を
指定します。詳細は [docs/multi-gpu.md](docs/multi-gpu.md) を参照してください。

## 性能

RTX 3090 × 2 での実測（2026-09-10、本計測 7 回の中央値、単位は ms）。

| 演算 | 入力 bit | tabai（1 GPU） | tabai（2 GPU） | GMP（CPU） |
| --- | ---: | ---: | ---: | ---: |
| 乗算 | 100,000,000 | 36.3 | 32.0 | 873.3 |
| 二乗 | 100,000,000 | 25.2 | 25.7 | 589.3 |
| 3 乗（`** 3`） | 100,000,000 | 104.8 | 91.2 | 1980.1 |
| 乗算 | 512,000,000 | 176.0 | 146.6 | — |

GPU のカーネル起動・同期・転送のコストがあるため、小さい入力では CPU の方が速いことがよくあります。
全結果は [docs/benchmarks/rtx3090-2026-09-10.md](docs/benchmarks/rtx3090-2026-09-10.md) にあります。

## テストとベンチマーク

```bash
# テスト
docker compose exec tabai pytest tests/ -q

# 全演算を 単一 GPU・GMP・Python int で比較
docker compose exec -T tabai python benchmarks/benchmark.py \
    --gpu-mode single --backends tabai gmpy2 python --bits 1000000 --pow-exponents 3

# 乗算のみを 単一 GPU と 2 GPU で比較
docker compose exec -T tabai python benchmarks/bench_mul.py \
    --gpu-mode both --devices 0 1 --backends tabai --bits 100000000
```

CPU バックエンドの計測には CUDA も Docker も不要です（`gmpy2` を使う場合はその Python 環境にインストールしてください）。

```bash
python3 tabai/benchmarks/benchmark.py --backends python gmpy2 --bits 1000 10000
```

全オプションと出力形式は [docs/benchmarks.md](docs/benchmarks.md) にまとめてあります。

## 実装の概要

| ファイル | 役割 |
| --- | --- |
| [tabai_int.py](tabai/src/tabai_gpu/tabai_int.py) | 公開クラス `TabaiInt`、符号管理、演算子、Python `int` との相互運用 |
| [core.py](tabai/src/tabai_gpu/core.py) | `GPUBigInt` と CUDA カーネル、作業バッファ・NTT テーブルの再利用 |
| [multi_gpu.py](tabai/src/tabai_gpu/multi_gpu.py) | NTT の GPU 間分割と `multi_gpu` コンテキストマネージャ |
| [utils.py](tabai/src/tabai_gpu/utils.py) | Python 整数と GPU 配列の相互変換 |

絶対値は下位桁から並ぶ `uint32` 配列として GPU に保持し、符号は Python 側で管理します。

- **加算・減算** — 並列プレフィックススキャンで桁上がり・桁借りを伝播します。
- **乗算** — 小さい入力は筆算方式、大きい入力は Goldilocks 素数 `p = 2^64 − 2^32 + 1` 上の NTT を使います。
  すべて剰余演算なので、浮動小数点の丸めは発生しません。
- **除算・剰余** — Newton–Raphson 法で逆数を求め、商を ±1 の範囲で補正します。
- **累乗** — 指数を CPU で読み取り、スライディングウィンドウ法で GPU の乗算・二乗を繰り返します。

公開クラスに未実装のもの：`/`、ビット演算・シフト、3 引数の `pow(a, b, mod)`、`int(a)`。
Python の整数へは `to_cpu()` で変換します。

## ドキュメント

- [docs/multi-gpu.md](docs/multi-gpu.md) — 複数 GPU のアルゴリズム、設定、検証手順
- [docs/benchmarks.md](docs/benchmarks.md) — ベンチマーク CLI のリファレンス
- [docs/performance.md](docs/performance.md) — 性能改善レポート
- [CLAUDE.md](CLAUDE.md) — 開発時の構成と実装上の注意点

## ライセンス

[MIT License](LICENSE)
