# 1 回の大きな整数演算を複数 GPU で処理する

`multi_gpu` コンテキスト内では、大きな乗算の NTT を GPU 間で分割します。`TabaiInt` の乗算演算子をそのまま使え、累乗と Newton 法による除算の内部乗算にも適用されます。加減算・比較・桁上がり処理・最終結果の保持は先頭の GPU で行います。

## 使用例

```python
import cupy as cp
from tabai_gpu import TabaiInt, multi_gpu

with cp.cuda.Device(0):
    a = TabaiInt((1 << 2_000_000) - 1)
    b = TabaiInt((1 << 1_000_000) + 7)
    with multi_gpu([0, 1], min_bits=0):  # この例では小さい入力でも分散を有効にする
        product = a * b
        power = a ** 3
        q, r = divmod(a, b)
    # 結果はコンテキスト終了後も GPU 0 上で利用できる。
    assert product.to_cpu() == ((1 << 2_000_000) - 1) * ((1 << 1_000_000) + 7)
```

指定できる GPU 枚数は `1, 2, 4, ...` の 2 冪です。GPU ID はプロセスから見える CUDA の番号で、重複や存在しない ID はエラーになります。`[0]` は分散コードを 1 枚で検証するためにも使えますが、複数枚での動作検証にはなりません。

呼び出し時のデバイス・入力配列・演算エンジンは `devices[0]` に揃えてください。`[1, 0]` を使う場合は `cp.cuda.Device(1)` 内で値を生成して演算します。異なる GPU に置いた `TabaiInt` 同士を自動で移動する API ではありません。

| 引数 | 既定値 | 意味 |
| --- | --- | --- |
| `devices` | `(0, 1)` | NTT の分割先。先頭が入力・出力を保持する GPU |
| `min_bits` | `64_000_000` | 長い方のオペランドの格納桁数 × 32 がこの値以上なら分散候補 |
| `transfer` | `"auto"` | P2P が使える方向は直接転送、使えない方向は CPU 経由。`"host"` は CPU 経由を強制 |

既存の筆算方式の条件が先に適用されるため、`min_bits=0` でも小さい乗算は分散しません。既定の 64 Mbit は当初の実装の RTX 3090 × 2、CPU 経由転送の計測をもとに選んでいます。小さい入力では分散による計算時間の削減より転送コストが大きかったためです。他の接続構成や二乗・非対称入力での最適値を保証するものではありません。コンテキストを使わない演算は従来の単一 GPU 経路を使用します。

## 分散方法

変換長 `n` の係数配列を、GPU 数 `D` に対して連続した `n/D` 要素ずつに分割します。

1. 入力を 16 bit チャンクにして各 GPU へ配布し、残りをゼロで埋めます。
2. DIF 順変換の先頭 `log2(D)` 段は、対応する GPU と係数を交換してバタフライを計算します。残りの段は各 GPU 内で独立に計算します。
3. 各 GPU で要素ごとの乗算と **全体の変換長に対する `1/n`** のスケーリングを行います。同一配列の二乗では順変換を 1 回に省略します。
4. DIT 逆変換を各 GPU 内で進め、最後の `log2(D)` 段で GPU 間の係数を交換します。
5. 線形畳み込みの係数を先頭の GPU に集め、既存の桁上がり処理で `uint32` の整数配列に戻します。

2 GPU では転送を減らすため、次の最適化を加えています。入力が変換配列の下半分に収まる場合、上半分はゼロなので、最初の DIF 段は `lower=u`、`upper=u*w` になります。両 GPU に 16 bit 入力を配布してこれを直接構成し、64 bit 係数とゼロ領域の交換を省きます。長い非対称入力では通常の交換に戻ります。逆変換の最終段は GPU 1 の部分結果だけを GPU 0 に集め、桁上がりバッファへの書き込みとまとめて処理します。

GPU ごとに独立した非同期ストリームと作業配列を持ちます。交換前に送信側の計算完了を待ち、すべての受信が完了してから入力を上書きします。P2P 転送にはストリームを指定できる CuPy の [MemoryPointer.copy_from_device_async](https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.MemoryPointer.html#cupy.cuda.MemoryPointer.copy_from_device_async) を使用します。P2P の可否は接続構成に依存します。[CuPy の複数デバイスに関する説明](https://docs.cupy.dev/en/stable/user_guide/basic.html#current-device)

CPU 経由の転送には再利用するページ固定メモリを使います。大きな連続配列は 2 MiB ごとに分け、専用ストリームでの GPU → CPU 読み出しと CPU → GPU 書き込みを重ねます。各チャンクの読み出し完了後にアップロードし、最後のアップロード完了イベントをバッファの再利用・拡張前に待ちます。各 GPU 内の NTT は、2 段の融合と最大 1,024 係数の共有メモリ処理で全配列の読み書きを減らしています。

通常の `TabaiInt` エンジンもデバイス・ストリーム別に遅延生成します。これにより別 GPU の定数や作業バッファを誤って使うことを避けます。同じコンテキスト／エンジンの複数スレッドからの同時利用はサポート対象にしていません。

## 制約と性能の読み方

- Goldilocks NTT の変換長は `2^32` 以下、列和は `2^63` 未満に制限します。さらに既存 CUDA カーネルの符号付き 32 bit 添字に収まることを、分散経路では配列確保前に検査します。
- NTT の作業配列は分割しますが、先頭 GPU には入力・結果と全体の桁上がりバッファが必要です。GPU メモリを単純に合算してすべて利用できる構成ではありません。
- GPU 間の交換と同期が増えます。特に CPU 経由の転送では通信が計算時間を上回る可能性があり、複数枚での高速化は実測が必要です。
- 作業配列と NTT テーブルはコンテキスト内で再利用します。終了時に分散エンジンの参照を解放しますが、CuPy のメモリプールが解放済みメモリを保持することはあります。
- 対象は同じホスト上の CUDA デバイスです。複数 VM 間の分散処理は実装していません。

## 性能改善後の検証（2026-09-10）

NTT と CPU 経由転送を改善し、全テストは **648 passed, 2 skipped** でした。1 億 bit の 2 GPU 乗算は改善前 32.7 ms から 19.0 ms に短縮しました。単一 GPU も 19.3 ms に改善しており、このサイズでは両者はほぼ同等です。512 Mbit の乗算は単一 GPU 99.2 ms、2 GPU 89.9 ms でした。64 Mbit の分散乗算・二乗は単一 GPU より遅いため、既定の分散開始サイズが最適とは限りません。計測条件・全結果・残る制約は [性能改善レポート](performance.md) を参照してください。

以下の 2026-09-07 の結果は改善前の履歴です。

## 実機検証結果（2026-09-07）

RTX 3090 24 GB × 2、ドライバー 580.173.02、CUDA 13.0、CuPy 14.0.1 で確認しました。接続は `PHB`、`deviceCanAccessPeer(0, 1)` と逆方向はいずれも false です。したがって以下は **CPU 経由転送での実測**で、物理 P2P 経路と実機 4 GPU は未検証です。

`pytest tests/ --require-multi-gpu -q` は **611 passed, 2 skipped**。スキップ 2 件は、重複 GPU ID を使う論理分割を公開 API テストの対象外にするケースです。実機 2 GPU、先頭デバイスの入れ替え、CPU 経由の強制転送は実行済みです。

既定の `min_bits=64_000_000`、ウォームアップ 2 回、本計測 5 回の中央値です。入力長 100,000,000 bit、除算の除数はその半分、累乗の指数は 3 です。

| 演算 | 単一 GPU | 2 GPU | 単一 GPU 時間 ÷ 2 GPU 時間 |
| --- | --- | --- | --- |
| 乗算 | 36.24 ms | 31.39 ms | 1.15× |
| 二乗 | 25.12 ms | 25.20 ms | 1.00× |
| 3 乗 | 104.81 ms | 88.75 ms | 1.18× |
| 除算 | 238.12 ms | 227.54 ms | 1.05× |

1 Mbit・10 Mbit の行は分散呼び出しが 0 で、単一 GPU 経路を維持しています。これらの速度比の微小差は分散による高速化ではありません。GMP と全計測結果を照合済みです。

100 Mbit 乗算の計測後に CuPy プールが予約していた量は、単一 GPU で GPU 0 が 964,734,464 byte、分散時は GPU 0 が 794,973,696 byte、GPU 1 が 335,546,368 byte でした。先頭 GPU の負担は減りますが、合計予約量は増えます。この値は同じプロセスで実行した前のサイズのキャッシュも含み、厳密なピークメモリではありません。

測定ログは [最終ベンチマーク JSON](benchmarks/multi-gpu-rtx3090.json) と [分散開始サイズの比較 JSON](benchmarks/multi-gpu-threshold-rtx3090.json) に保存しています。閾値比較は `--min-bits 0` で 1/4/16/32/64 Mbit の乗算・二乗を測定したものです。

## GPU を 2 枚に載せ替えた後の検証

Compose は既に `count: all` で GPU を公開する設定です。VM で 2 枚が見えることを確認し、コンテナを再作成して実機テストを実行します。

```bash
nvidia-smi -L
nvidia-smi topo -m
docker compose up -d --force-recreate
docker compose exec -T tabai python -c 'import cupy as cp; print(cp.cuda.runtime.getDeviceCount())'
docker compose exec -T tabai pytest tests/test_multi_gpu.py --require-multi-gpu -q
docker compose exec -T tabai pytest tests/ --require-multi-gpu -q
```

`--require-multi-gpu` は 2 枚未満の場合にエラー終了するため、実機テストをスキップしただけの成功を防げます。テストには直接転送の自動選択と CPU 経由の強制転送、先頭 GPU を 1 に切り替える場合、負数・累乗・除算、非既定ストリーム、二乗・長い桁上がり・サイズ変更を含めています。

単一 GPU と分散演算を、**同一入力・別プロセス**で比較します。

```bash
docker compose exec -T tabai python benchmarks/bench_multi_gpu.py --devices 0 1 --output /mnt/multi-gpu-auto.json
docker compose exec -T tabai python benchmarks/bench_multi_gpu.py --devices 0 1 --transfer host --output /mnt/multi-gpu-host.json
```

現在のスクリプトは全演算共通の CLI を使います。新しい JSON 形式とエラー処理は [ベンチマークの使い方](benchmarks.md) を参照してください。

出力 JSON はホストの `tabai/multi-gpu-auto.json` / `tabai/multi-gpu-host.json` に保存されます。標準出力には乗算・二乗・3 乗・除算の中央値、速度比（単一 GPU 時間 ÷ 分散時間）、分散 NTT の呼び出し回数を表示します。JSON には各試行時間、GPU 名・ドライバー・CuPy バージョン、P2P 可否、転送回数、各行の計測後に CuPy メモリプールが予約しているバイト数を含めます。このメモリ量は CUDA コンテキストなどを含む GPU 全体の厳密なピーク使用量ではありません。

入力転送と結果の CPU 変換は計測区間外、**分散に必要な GPU 間転送と同期は計測区間内**です。GMP と各試行結果を照合し、不一致や実行エラーはそのまま失敗として扱います。ウォームアップ 2 回、本計測 5 回が既定です。まず短く確認する場合は次を使えます。

```bash
docker compose exec -T tabai python benchmarks/bench_multi_gpu.py --devices 0 1 --bits 1000000 --min-bits 0 --repeat 2
```

`NTT calls` が 0 の行は分散開始サイズを下回るなどの理由で単一 GPU 経路を使用しています。分散そのものを小さい入力で比較する場合は `--min-bits 0` を指定してください。

1 枚の段階では `--gpu-mode single --devices 0` で単一 GPU の計測を実行できます。実機 2 枚での転送・同時実行・高速化の検証結果とは区別してください。

## この VM で確認した CUDA 起動エラー

コンテナ内の互換性用 `libcuda` がホストから渡されたドライバーより優先され、RTX 3090 で `cudaErrorCompatNotSupportedOnDevice` が発生しました。この VM では、対応済みのホスト側ライブラリのディレクトリを優先すると GPU テストを実行できました。必要な場合は `docker compose exec` に以下の環境指定を追加します。

```bash
docker compose exec -T -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64 tabai pytest tests/test_multi_gpu.py -q
```

これはこの VM のライブラリ配置に対する指定です。エラー 804 と対応ハードウェアの条件は [NVIDIA の CUDA 互換性ドキュメント](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html) を参照してください。
