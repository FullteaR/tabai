# NTT 移行手順書 — `GPUBigInt.mul` の FFT → NTT 置き換え

> **現在の位置づけ**: この文書は移行時の設計・作業計画です。現在の `core.py` は大サイズ乗算に `_mul_ntt` を使用し、旧 `_mul_fft` は撤去されています。筆算方式との閾値も `6016 * 6016` に更新済みです。以下の「現状」、旧識別子、フェーズ指示は移行前の記録として残しています。現在の使い方と制約は [README.md](../README.md) を参照してください。コードの状態だけでは、本文中の性能合格基準や全フェーズの完了を確認できません。

## 0. この文書について

- **読者**: 実装を担当する AI エージェント(Claude Opus 4.8 を想定)。
- **目的**: `tabai/src/tabai_gpu/core.py` の大サイズ乗算パス `_mul_fft`(float64 の `cp.fft.rfft` ベース)を、Goldilocks 素数上の NTT(数論変換)に置き換える。
- **進め方**: 本文書の Phase 0 → 4 を順に実行する。各 Phase の「Done 条件」を満たしてから次へ進む。Phase ごとに 1 コミット以上(prefix は既存履歴に合わせ `feat:` / `perf:` / `test:` / `refactor:`)。
- **§2 の設計決定と §3 の数学は確定事項**。本文書作成時に Python で網羅的に検証済み(素数性、原始根、還元アルゴリズム、DIF/DIT ペアの畳み込み正当性をランダム 20 万ケース以上で確認)。再設計・再選定はしない。ただし Phase 0 で検証スクリプトを自分でも再実行して確認すること。
- **禁止事項**:
  - `TabaiInt` の公開 API・符号処理・`_resolve_carries` / `_recombine_chunks` / schoolbook パスの変更(再利用のみ)。
  - 既存テストの弱体化(期待値の緩和・削除)。閾値変更に伴う境界テストの定数更新は可。
  - 本移行と無関係なリファクタリング。
- **中断条件**: §6 の合格基準を Phase 3 の最適化後も満たせない場合は、マージせずベンチ結果と分析を添えて報告し停止する。

## 1. 背景とゴール

### 現状

`GPUBigInt.mul` は operand の仕事量 `la*lb` で分岐する:

- `la*lb <= _MUL_SCHOOLBOOK_MAX_WORK (5120²)` → `_mul_schoolbook`(base-2^16 列和 + 共有キャリー解決パイプライン)
- それ以上 → `_mul_fft`: uint32 limb を uint16/uint8 チャンクに分解 → `cp.fft.rfft` で畳み込み → `cp.rint` で丸め → int64 列和として `_resolve_carries` → `_recombine_chunks`

FFT パスの問題点:

1. **float64 の丸め誤差リスク**: IFFT 係数が 2^53 を超えないようチャンク幅を適応制御している(n_fft ≥ 2^20 で 16bit → 8bit に退行)。正当性が「誤差が閾値内に収まるはず」という確率的な議論に依存し、極端なサイズでの正確性は保証されない(CLAUDE.md の Key Constraints にも明記されている既知の制約)。
2. **8bit チャンク退行のコスト**: ~4 Mbit 超で変換長が 2 倍、float64 バッファも 2 倍になる。

### NTT にすると

- 全演算が mod p の整数演算になり、**任意サイズで数学的に正確**(丸めなし、`cp.rint` 不要)。
- **チャンク幅は常に 16bit** で済む(境界導出は §3.4)。大サイズで FFT 比の変換長が半分。
- 引き換えに、cuFFT という高度に最適化されたライブラリを、自作 radix-2 カーネルで置き換えることになる。中規模帯(~160 Kbit – 4 Mbit)では初期実装は FFT より遅い可能性が高い。§6 の基準と最適化手順で対処する。

### ゴール

1. `mul` の大サイズパスが NTT になり、全サイズで正確(既存 + 新規テスト全 green)。
2. ベンチマーク(§6)で FFT ベースライン比の許容範囲内。
3. FFT コードの完全撤去とドキュメント更新。

### 非ゴール

- add/sub/divmod/pow のアルゴリズム変更(divmod・pow は mul 経由で自動的に恩恵を受ける)。
- multi-GPU、stream 並列化、Karatsuba 等の別アルゴリズム。

## 2. 設計決定(確定)

### 2.1 法: Goldilocks 素数 1 本

```
p = 2^64 − 2^32 + 1 = 18446744069414584321 = 0xFFFFFFFF00000001
```

- `p − 1 = 2^32 · (2^32 − 1)` なので 2-adicity は 32。**2^32 までの任意の 2 冪長の変換が可能**(GPU メモリの方が先に尽きる)。
- `7` は p の原始根。サイズ n の原始 n 乗根は `pow(7, (p−1)//n, p)` で導出する。
- 128bit → 64bit の還元が `2^64 ≡ 2^32 − 1`, `2^96 ≡ −1 (mod p)` を使って分岐 2 回・乗算 1 回程度で書ける(§3.2)。Montgomery / Barrett は不要。
- 係数は uint64 1 本に収まり、既存の int64 キャリーパイプラインにそのまま流せる。

**棄却した代替案**(再検討しないこと):

- *31/32bit 素数 3 本 + CRT*: 1 素数あたりの modmul は安いが、変換 3 セット + CRT 再結合が必要でメモリ帯域・複雑性ともに不利。32bit チャンク化して変換長を半分にする案は列和が 2^93 に達し int64 キャリーパイプラインが使えなくなるため不成立。
- *Goldilocks + 32bit チャンク*: 係数上限 `min_chunks · (2^32−1)² < p` より min_chunks < 4 で不成立。16bit が唯一の正解。

### 2.2 チャンク幅: 16bit 固定

uint32 limb を little-endian の uint16 ペアに view する(既存 FFT パスの 16bit 分岐と同一)。適応チャンク制御(8bit フォールバック)は丸ごと消える。上限は §3.4 参照 — 実効的に「GPU メモリに載る限り安全」。防御的 assert を 1 本入れる。

### 2.3 変換アルゴリズム: radix-2、DIF 順変換 / DIT 逆変換(ビット反転なし)

- **順変換**: Gentleman–Sande (DIF) バタフライ。自然順入力 → ビット反転順出力。
- **逆変換**: Cooley–Tukey (DIT) バタフライを ω⁻¹ で。ビット反転順入力 → 自然順出力。
- 中間で行うのは要素ごとの積だけなので順序は不問。**ビット反転並べ替えカーネルは不要**(競プロの ACL convolution と同じ構成)。
- **ステージごとに 1 カーネル launch**(n/2 スレッド、in-place)。ステージ間は launch 境界がグローバル同期を兼ねる。1 launch 内で複数ステージを回そうとしない(Phase 3 の shared-memory 融合を除く)。
- **twiddle 表**: サイズ n につき `w_fwd[k] = ω^k`, `w_inv[k] = ω^{−k}` (k < n/2) の 2 本を uint64 で GPU 上に持つ。ステージ(span m)では stride `n/m` で引く: `w[j * (n/m)]`。表の生成は per-thread binary powmod カーネル(§4 Phase 1)。サイズごとに dict でキャッシュ(cuFFT の plan cache に相当)。
- **逆変換後の 1/n スケーリングは pointwise 乗算カーネルに融合**: `c[k] = a[k]·b[k]·inv_n mod p`。忘れると結果が n 倍になる(最頻出バグ)。

### 2.4 統合方針

- `_mul_ntt(a_gpu, b_gpu, is_square=False)` を `_mul_fft` と同一シグネチャ・同一契約で実装: uint32 limb 配列 2 本 → uint32 limb 配列(長さ ≤ la+lb)。
- NTT 出力(= 正確な base-2^16 列和、uint64)を int64 キャリーバッファに書き、**既存の `_resolve_carries` → `_recombine_chunks` をそのまま呼ぶ**。`max_bits` は schoolbook パスと同じ式: `2*chunk_bits + min_chunks.bit_length()`。
- `is_square`(`a_gpu is b_gpu`)のとき順変換 1 回 + pointwise 自乗。ディスパッチ側の判定は既存のまま。
- schoolbook との閾値 `_MUL_SCHOOLBOOK_MAX_WORK` は Phase 3 で再スイープ。
- Phase 2〜3 の間は `_mul_fft` を残して A/B 比較に使い、Phase 4 で撤去する。

## 3. 検証済み数学リファレンス

このセクションのコードは本文書作成時に Python で検証済み。CUDA 化の際にロジックを変えないこと。

### 3.1 定数と自己検証

```python
P = (1 << 64) - (1 << 32) + 1          # 0xFFFFFFFF00000001, prime
# p−1 = 2^32 · 3 · 5 · 17 · 257 · 65537  (2^32 − 1 = 3·5·17·257·65537)
for q in [2, 3, 5, 17, 257, 65537]:
    assert pow(7, (P - 1) // q, P) != 1   # 7 is a primitive root
assert pow(7, P - 1, P) == 1
w32 = pow(7, (P - 1) >> 32, P)
assert w32 == 0x185629DCDA58878C          # cross-check (Plonky2 と同一値)
assert pow(w32, 1 << 31, P) == P - 1      # order exactly 2^32
```

サイズ n = 2^k 用の根: `w_n = pow(7, (P-1)//n, P)`、逆根: `pow(w_n, P-2, P)`、`inv_n = pow(n, P-2, P)`。すべて Python の `pow` で正確に計算し GPU に渡す。

### 3.2 mod p 演算(CUDA device 関数)

以下のロジックはエッジ値(0, 1, 2^32−1, 2^32, 2^63, p−2, p−1)の全組合せ + ランダム 20 万ペア、および任意 128bit 入力 5 万件で `(a*b) % P` と一致することを検証済み。**このまま実装する**:

```cuda
#define GL_P   18446744069414584321ULL   /* 2^64 - 2^32 + 1 */
#define GL_EPS 0xFFFFFFFFULL             /* 2^32 - 1  ( = 2^64 mod p) */

/* reduce a 128-bit value hi*2^64 + lo into [0, p).
   Uses 2^64 ≡ 2^32 − 1 and 2^96 ≡ −1 (mod p):
     x ≡ lo + (2^32−1)*h0 − h1,  where hi = h1*2^32 + h0.          */
__device__ __forceinline__ unsigned long long gl_reduce128(
        unsigned long long hi, unsigned long long lo)
{
    unsigned long long h0 = hi & GL_EPS;
    unsigned long long h1 = hi >> 32;
    unsigned long long t  = lo - h1;
    if (lo < h1) t -= GL_EPS;            /* borrow: wrapped by +2^64 ≡ +EPS */
    unsigned long long u = h0 * GL_EPS;  /* < 2^64, exact */
    unsigned long long r = t + u;
    if (r < t) r += GL_EPS;              /* carry: wrapped by −2^64 ≡ −EPS */
    if (r >= GL_P) r -= GL_P;
    return r;
}

__device__ __forceinline__ unsigned long long gl_mulmod(
        unsigned long long a, unsigned long long b)
{
    return gl_reduce128(__umul64hi(a, b), a * b);
}

/* inputs must be canonical (< p); outputs are canonical */
__device__ __forceinline__ unsigned long long gl_addmod(
        unsigned long long a, unsigned long long b)
{
    unsigned long long s = a + b;
    if (s < a) s += GL_EPS;
    if (s >= GL_P) s -= GL_P;
    return s;
}

__device__ __forceinline__ unsigned long long gl_submod(
        unsigned long long a, unsigned long long b)
{
    unsigned long long s = a - b;
    if (a < b) s -= GL_EPS;
    return s;
}
```

`gl_reduce128` は**任意の** 128bit 入力に対して正しい(入力が canonical でなくてもよい)。`gl_addmod`/`gl_submod` は canonical 入力前提 — バッファは常に canonical に保たれる(初期値は uint16 チャンク < 2^16 < p、以後の全演算が canonical を返す)ので問題ない。

### 3.3 参照 NTT(Python、テスト用オラクル)

小サイズの GPU 実装検証と `tests/test_ntt.py` に使う。350 ランダムケースで `a*b`(Python int)との一致を検証済み:

```python
def ntt_dif(a, w, p):                 # natural order in -> bit-reversed out
    n = len(a); m = n
    while m >= 2:
        half = m // 2; stride = n // m
        for base in range(0, n, m):
            for j in range(half):
                u, v = a[base + j], a[base + j + half]
                a[base + j] = (u + v) % p
                a[base + j + half] = (u - v) * w[j * stride] % p
        m = half

def ntt_dit_inv(a, winv, p):          # bit-reversed in -> natural order out
    n = len(a); m = 2
    while m <= n:
        half = m // 2; stride = n // m
        for base in range(0, n, m):
            for j in range(half):
                u = a[base + j]
                t = a[base + j + half] * winv[j * stride] % p
                a[base + j] = (u + t) % p
                a[base + j + half] = (u - t) % p
        m *= 2
```

畳み込み全体: `n = 次の2冪(len(a_ch)+len(b_ch)−1)` に zero-pad → 両者 `ntt_dif` → `c[k] = a[k]*b[k]*inv_n % p` → `ntt_dit_inv(c)` → 先頭 `n_conv` 係数が正確な列和。

### 3.4 係数境界(なぜ 16bit チャンクが常に安全か)

列和の最大値は `min_chunks × (2^16−1)²`(min_chunks = 短い方のオペランドの 16bit チャンク数)。

- NTT の正確性条件(< p): min_chunks ≤ 4,295,098,370 ≈ **2^32**
- int64 キャリーパイプライン条件(< 2^63): min_chunks ≤ 2,147,549,185 ≈ **2^31**(こちらが束縛。オペランド ~32 Gbit 相当)
- 変換長条件: n_fft ≤ 2^32(2-adicity)

RTX 3090 (24 GB) では n_fft ≈ 2^28 前後でメモリが先に尽きるため、実質無制限。`_mul_ntt` 冒頭に防御的 assert を入れる:

```python
assert min(n_a16, n_b16) * 0xFFFF**2 < (1 << 63)
```

## 4. 実装フェーズ

作業はすべてリポジトリルート(ホスト)から Docker 経由で行う(§9 コマンド集)。開始前に `docker compose up -d` でコンテナが起動していることを確認。

---

### Phase 0 — 準備・ベースライン・参照実装

1. ブランチ `feat/ntt-mul` を main から作成。
2. 現状確認: `pytest tests/` が全 green であること(壊れていたら報告して停止)。
3. **ベースライン計測**: `benchmarks/bench_mul.py`, `bench_pow.py`, `bench_div.py` を実行し、出力をホスト側(リポジトリ外、例 `/tmp/baseline-*.txt`)に保存。タイムアウト/OOM の行(`>timeout`, `--`)もそのまま記録 — 移行後も同条件比較する。
4. `tests/test_ntt.py` を新規作成し、まず GPU に依存しない部分を入れる:
   - §3.1 の定数自己検証テスト(p の素数性は 2^64 未満用の決定的 Miller–Rabin、基底 [2,3,5,7,11,13,17,19,23,29,31,37])。
   - §3.3 の参照 NTT をテストヘルパとして実装し、ランダムな Python int ペア(~数千 bit)で `a*b` と一致することを確認するテスト。

**Done 条件**: 既存テスト + 新テスト全 green。ベースライン 3 表が保存済み。

---

### Phase 1 — GPU NTT 基盤と `_mul_ntt`(まだ配線しない)

`core.py` に追加する(既存のモジュールレベル `cp.RawKernel` 定義スタイルに合わせる)。

**1. 共通プリアンブル**: §3.2 の device 関数群を 1 つの文字列定数(例 `_GL_PREAMBLE`)にし、各 NTT カーネルのソース先頭に連結する。定数 p / EPS の定義箇所を 1 か所にする。

**2. カーネル 4 本**:

- `gl_fill_powers(w, base, m)` — スレッド k (< m) が `w[k] = base^k mod p` を k のビットに対する square-and-multiply(`gl_mulmod` 使用、最大 31 ビット)で計算。twiddle 表生成用。
- `ntt_dif_stage(a, w, n_half, half, log_half, tw_stride)` — スレッド idx (< n_half):

  ```
  j  = idx & (half - 1);
  i0 = ((idx >> log_half) << (log_half + 1)) | j;
  i1 = i0 + half;
  u = a[i0]; v = a[i1];
  a[i0] = gl_addmod(u, v);
  a[i1] = gl_mulmod(gl_submod(u, v), w[(size_t)j * tw_stride]);
  ```

- `ntt_dit_inv_stage(a, winv, n_half, half, log_half, tw_stride)` — 同じ添字計算で:

  ```
  t = gl_mulmod(a[i1], winv[(size_t)j * tw_stride]);
  a[i0] = gl_addmod(u, t);
  a[i1] = gl_submod(u, t);
  ```

- `ntt_pointwise_scale(a, b, scale, n)` — `a[k] = gl_mulmod(gl_mulmod(a[k], b[k]), scale)`。in-place(a に書き戻し)。`b` に `a` 自身を渡す自乗も可(読み取りのみなのでエイリアス安全)。

添字・stride の積は `size_t` / 64bit で計算する(n_half が大きいときの int オーバーフロー防止)。launch は既存流儀どおり `blocks = (n_half + _BLOCK - 1) // _BLOCK`, `(_BLOCK,)` スレッド。

**3. ホスト側**:

- `_ensure_ntt_capacity(n)` — `_ensure_fft_capacity` に倣い、`_ntt_buf_a` / `_ntt_buf_b`(`cp.uint64`, 長さ n)を lazily 成長 + `_ensure_carry_capacity(n + 1)`。
- `_get_ntt_tables(n)` — dict キャッシュ。ミス時: Python で `w_n = pow(7, (P-1)//n, P)`, `w_n_inv = pow(w_n, P-2, P)`, `inv_n = pow(n, P-2, P)` を計算し、`gl_fill_powers` で `w_fwd`, `w_inv`(各長さ `max(1, n//2)`)を GPU 上に生成。`(w_fwd, w_inv, inv_n)` を返す。
- `_ntt_forward(a, n, w_fwd)` — `half = n>>1` から 1 まで(`tw_stride = (n>>1) // half` … 検算: 先頭ステージ m=n で stride 1、最終ステージ m=2 で stride n/2。§3.3 の `n//m` と同値)ステージごとに `ntt_dif_stage` を launch。
- `_ntt_inverse(a, n, w_inv)` — `half = 1` から `n>>1` まで逆順に `ntt_dit_inv_stage`。
- `_mul_ntt(a_gpu, b_gpu, is_square=False)`:

  ```python
  chunk_bits = 16
  a_ch = cp.ascontiguousarray(a_gpu).view(cp.uint16)
  b_ch = a_ch if is_square else cp.ascontiguousarray(b_gpu).view(cp.uint16)
  n_conv = len(a_ch) + len(b_ch) - 1
  n = 1 << max(0, n_conv - 1).bit_length()   # next pow2 >= n_conv (n_conv=1 → n=1)
  # §3.4 の防御的 assert
  w_fwd, w_inv, inv_n = self._get_ntt_tables(n)
  self._ensure_ntt_capacity(n)
  fa = self._ntt_buf_a[:n]; fa[:len(a_ch)] = a_ch; fa[len(a_ch):] = 0
  self._ntt_forward(fa, n, w_fwd)
  if is_square:
      fb = fa
  else:
      fb = self._ntt_buf_b[:n]; fb[:len(b_ch)] = b_ch; fb[len(b_ch):] = 0
      self._ntt_forward(fb, n, w_fwd)      # forward は b 側も w_fwd(w_inv は逆変換専用)
  # pointwise + inv_n 融合(is_square なら fb is fa)
  ntt_pointwise_scale(fa, fb, np.uint64(inv_n), n)
  self._ntt_inverse(fa, n, w_inv)
  carry_n = n + 1
  self._ensure_carry_capacity(carry_n)       # 必ず slice を取る前に
  ping = self._fft_carry_ping[:carry_n]
  ping[:n] = fa                              # uint64 -> int64 cast; < 2^62 なので安全
  ping[n] = 0                                # carry-out slot
  max_bits = 2 * chunk_bits + min(len(a_ch), len(b_ch)).bit_length()
  ping = self._resolve_carries(carry_n, chunk_bits, max_bits)
  return self._recombine_chunks(ping, carry_n, chunk_bits, len(a_gpu) + len(b_gpu))
  ```

  (疑似コード。n=1 のとき forward/inverse のステージループは空で正しく恒等になる。)

**4. `tests/test_ntt.py` に GPU テストを追加**(§5 の 1–4)。

**Done 条件**: 新テスト全 green(この時点で `mul()` はまだ FFT のまま)、既存テストも green。

---

### Phase 2 — 配線切り替え

1. `mul()` の `_mul_fft` 呼び出しを `_mul_ntt` に変更(`is_square=a_gpu is b_gpu` 維持)。`_mul_fft` 自体は A/B 用に残す。
2. 全テスト実行: `pytest tests/`。divmod / pow / boundary テストが mul 経由で NTT を叩く実質的な統合テストになる。
3. 簡易 A/B スクリプト(リポジトリ外)で `_mul_fft` vs `_mul_ntt` を同一入力・複数サイズ(160 Kbit〜1 Gbit、warmup 2 / repeat 5、`cp.cuda.Stream.null.synchronize()` で計時)で比較し、結果を記録。

**Done 条件**: 全 suite green。A/B 表が手元にある。

---

### Phase 3 — ベンチマークと調律

1. `benchmarks/bench_mul.py` / `bench_pow.py` / `bench_div.py` を再実行し、Phase 0 のベースラインと突き合わせて §6 の基準を判定。
2. **schoolbook ↔ NTT 閾値の再スイープ**: 平方オペランド L ∈ {3072 … 8192, 512 刻み} で `_mul_schoolbook` と `_mul_ntt` を直接計時し、交点 L* を求めて `_MUL_SCHOOLBOOK_MAX_WORK = L*²` に更新。既存コメントのスタイル(スイープ根拠を書く)を踏襲。`tests/test_boundary_conditions.py` の `_MUL_SCHOOLBOOK_L` / `_MUL_FFT_L` 系定数と番号・コメントを追随させる(名前は `_MUL_NTT_L` に)。
3. 基準未達の帯域があれば、まず **shared-memory ステージ融合** を実装する: DIF の終盤(half が小さい)/ DIT の序盤ステージは、1 ブロックが 2·blockDim 要素を shared memory にロードして複数ステージを `__syncthreads()` 区切りで回せる(グローバル往復が log 回 → 1 回に減る、最大の効果)。大 stride ステージは per-launch のまま。radix-4 化はその次の手段。
4. (任意・時間制限つき)`_DIV_NEWTON_THRESHOLD_LIMBS` の交点再確認。mul が速くなった/遅くなった分だけずれる可能性がある。ずれが小さければ据え置きで PR に注記。

**Done 条件**: §6 の合格基準を満たす(満たせなければ §0 中断条件へ)。

---

### Phase 4 — FFT 撤去とクリーンアップ

1. `_mul_fft`、`_fft_buf_a/b`、`_ensure_fft_capacity`、`cp.rint` 呼び出し、適応チャンク幅ロジックを削除。
2. リネーム: `_fft_carry_ping/pong` → `_carry_ping/pong`、`_ensure_fft_capacity` 系コメントの整理。schoolbook パスと `_resolve_carries` 内の参照も追随。
3. コメント更新: `_carry_prop_step_kernel` ヘッダの「FFT-based multiplication」、`_NEWTON_RAMP_TRIM_LIMBS` の「FFT transform-length boundary」(NTT でも 2 冪境界のコスト崖は同じなので「transform-length boundary」に言い換え。値は据え置き可)。
4. `CLAUDE.md` 更新: Layer 1 の Mul 説明を NTT(Goldilocks p、16bit チャンク、DIF/DIT、正確性保証)に書き換え、Key Constraints から float64 精度制約の項を削除し「mul は mod-p NTT で全サイズ正確」に置き換える。
5. 最終確認: `grep -ri fft tabai/src/` がヒット 0 であること。全テスト + 全ベンチ再実行。
6. PR 作成(§10 のフォーマット)。

**Done 条件**: grep クリーン、全 suite green、PR に before/after ベンチ表。

## 5. テスト計画(`tests/test_ntt.py`)

既存 3 suite(`test_operations.py`, `test_tabai_int.py`, `test_boundary_conditions.py`)が最終的なオラクル。それに加えて:

1. **定数自己検証**(Phase 0): p の素数性(決定的 MR)、原始根 7、`w32 == 0x185629DCDA58878C`、`pow(w32, 2^31, p) == p−1`。
2. **`gl_mulmod` カーネル検証**: `ntt_pointwise_scale` を `scale=1` で流用し、エッジ値 {0, 1, 2, 2^32−1, 2^32, 2^63, p−2, p−1} の全ペア + ランダム uint64(< p)1 万ペアで Python の `(a*b) % p` と一致。**還元バグはここで全て捕まえる**(下流デバッグは困難)。
3. **往復恒等**: n ∈ {1, 2, 4, …, 4096} でランダム係数(< p)に forward → inverse → `inv_n` スケール(pointwise を全 1 配列と `scale=inv_n` で流用)して元に戻ること。
4. **参照実装一致**: n ∈ {8, 64, 1024} で GPU forward の出力が §3.3 の Python `ntt_dif` と全要素一致(順変換単体の検証。ここが合えばステージ添字は正しい)。
5. **`_mul_ntt` 直接テスト**: ランダム int 積との一致。
   - チャンク数が 2 冪境界を跨ぐサイズ(n_conv が 2^k−1, 2^k, 2^k+1 になる limb 数)
   - 非対称オペランド(1 limb × 10^5 limb 等)
   - 自乗パス(同一配列を 2 回渡し `is_square=True`)
   - all-ones パターン `(2^N − 1)²`(最悪キャリー、既存 `test_mul_all_ones_stress` の NTT 版)
6. **大サイズ正確性**(旧 FFT が 8bit 退行していた領域): 8 Mbit・64 Mbit のランダム積を Python int と照合(1〜2 ケースで十分。`@pytest.mark.slow` 等は既存に前例がないので普通のテストでよいが、実行時間 > 30 秒なら件数を絞る)。
7. **エンジン再利用**: 同じ `GPUBigInt` インスタンスで異なるサイズの mul を交互に実行(バッファ再利用・pad 消去のステールデータバグ検出)。

## 6. ベンチマークと合格基準

計測は §9 のコマンドで、ベースラインと同一マシン(RTX 3090)・同一手順。

| 項目 | 基準 |
|---|---|
| 正確性 | 既存 + 新規テスト全 green(**必須・妥協不可**) |
| `bench_mul` | ベースラインで完走した全行で NTT ≤ 1.5× FFT 時間(目標 ≤ 1.0×)。1 Gbit 級では ≤ 1.0× を強く期待(FFT は 8bit 退行帯) |
| `bench_pow` / `bench_div` | 全行で ≤ 1.3×(mul 支配なので連動する) |
| メモリ | ベースラインで完走したサイズが OOM にならないこと |

判定ルール: 未達行があれば Phase 3-3 の最適化(shared-memory 融合 → radix-4)を先に実施。それでも未達なら中断してデータと共に報告(FFT を温存する 3-way dispatch は本人の判断で導入しない — 報告して指示を仰ぐ)。

初回呼び出しは twiddle 表生成を含む(cuFFT の plan 生成と同様)。`bench()` の warmup 2 回で除外されるので特別扱い不要。10 Gbit 行はベースライン同様 OOM/timeout で構わない。

## 7. 落とし穴チェックリスト(実装中に随時参照)

1. **`inv_n` スケーリング忘れ** → 結果が n 倍。pointwise に融合し、往復恒等テストで検出。
2. **RawKernel への uint64 スカラー渡し**: Python int を渡さない。`np.uint64(inv_n)` を明示(ABI 不一致で無言のゴミ値になる)。ポインタ引数の順序も宣言と一致させる。
3. **`_ensure_carry_capacity` を呼ぶ前に `_fft_carry_ping` を slice しない**(再確保で古いバッファへの view が無効データになる)。既存 FFT パスの順序を踏襲。
4. **pad の消去**: `fa[len:] = 0` を必ず。バッファ再利用なので前回の値が残る(§5-7 のテストで検出)。
5. **uint64 → int64 代入**: 係数 < 2^62 なので cast 安全。ただし §3.4 の assert を先に置く。
6. **in-place ステージの並列性**: 1 スレッドがペア (i0, i1) を排他所有するので同一ステージ内は race なし。ステージ間の同期は launch 分割で担保 — 1 カーネルで複数ステージを回さない。
7. **添字の 64bit 化**: `j * tw_stride` や i0 の計算は `size_t`。n_half > 2^31 は現実的メモリでは起きないが、型は最初から安全側に。
8. **twiddle stride の検算**: 先頭 DIF ステージ(m=n)で stride=1・j < n/2 全域、最終ステージ(m=2)で stride=n/2・j=0(w[0]=1)。逆変換は逆順。§3.3 の Python 実装と 1 ステージずつ突き合わせられる。
9. **`ascontiguousarray` を省かない**: Newton 経路から来る配列は slice 由来のことがある。既存 FFT パスと同じ防御。
10. **カーネル内定数の一元化**: p / EPS はプリアンブル 1 か所で `#define`。手打ちの 20 桁定数を複数箇所に書かない。
11. **`is_square` のエイリアス**: pointwise で `fb is fa` でも読み取りのみなので安全。逆に DIF を 2 回目に呼ばないこと(1 回で済むのが最適化の趣旨)。
12. **ベンチの GPU 同期**: 自作 A/B 計測でも `cp.cuda.Stream.null.synchronize()` を挟む(`benchmarks/common.py` の `bench()` を手本に)。

## 8. 変更対象ファイル一覧

| ファイル | 変更 |
|---|---|
| `tabai/src/tabai_gpu/core.py` | NTT プリアンブル + カーネル 4 本、`_ensure_ntt_capacity` / `_get_ntt_tables` / `_ntt_forward` / `_ntt_inverse` / `_mul_ntt` 追加。`mul()` 配線変更。Phase 4 で FFT 撤去・リネーム・コメント更新 |
| `tabai/tests/test_ntt.py` | 新規(§5) |
| `tabai/tests/test_boundary_conditions.py` | 閾値定数・名称(`_MUL_FFT_L` → `_MUL_NTT_L`)の追随のみ |
| `CLAUDE.md` | Mul の説明と Key Constraints 更新(Phase 4) |
| その他 | 変更しない(`tabai_int.py`, `utils.py`, `benchmarks/` はそのまま) |

## 9. コマンド集(ホスト側リポジトリルートから)

```bash
docker compose up -d                                   # コンテナ起動(既に起動済みのはず)
docker compose exec tabai pytest tests/ -x -q          # 全テスト
docker compose exec tabai pytest tests/test_ntt.py -q  # NTT テストのみ
docker compose exec tabai python benchmarks/bench_mul.py   # mul ベンチ(pow/div も同様)
docker compose exec tabai python - <<'EOF'             # アドホック検証の型
from tabai_gpu.core import GPUBigInt
from tabai_gpu.utils import int_to_gpu, gpu_to_int
...
EOF
```

環境: コンテナ内 `PYTHONPATH=/mnt/src`、GPU は RTX 3090 (24 GB, CC 8.6)、CuPy v14。

## 10. 完了報告(PR)フォーマット

- タイトル例: `perf: replace float-FFT multiplication with exact Goldilocks NTT`
- 本文に含めること:
  1. 設計サマリ(法・チャンク幅・アルゴリズム、本文書へのリンク)
  2. before/after ベンチ表(mul / pow / div、ベースラインとの比率列つき)
  3. 閾値変更(`_MUL_SCHOOLBOOK_MAX_WORK` 等)とスイープ根拠
  4. 正確性の主張: 「float64 丸めへの依存を撤廃、全サイズで数学的に正確」+ テスト一覧
  5. 残課題(shared-memory 融合を実施した/しなかった、`_DIV_NEWTON_THRESHOLD_LIMBS` の扱い、等)
