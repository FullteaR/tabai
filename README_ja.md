# tabai

GPU を活用した Python 向け任意精度整数演算ライブラリ。

CUDA による NVIDIA GPU 並列処理を利用し、巨大な整数の加算・減算・乗算・除算を高速に実行します。

## 必要環境

- Docker
- CUDA 対応の NVIDIA GPU

## セットアップ

```bash
docker-compose build
docker-compose up -d
docker-compose exec tabai bash
```

## 使い方

```python
from tabai_gpu import TabaiInt

a = TabaiInt(1)
b = TabaiInt(2)

c = a + b
print(c.to_cpu())
#>> 3
```

### 四則演算

```python
from tabai_gpu import TabaiInt

a = TabaiInt(10 ** 100)
b = TabaiInt(20 ** 100)

# 加算
c = a + b
print(c.to_cpu())
#>> 12676506002282294014967032053770000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

# 減算
d = b - a
print(d.to_cpu())
#>> 12676506002282294014967032053750000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

# 乗算
e = TabaiInt(123456789) * TabaiInt(987654321)
print(e.to_cpu())
#>> 121932631112635269

# 整数除算
f = TabaiInt(100) // TabaiInt(3)
print(f.to_cpu())
#>> 33

# 剰余
g = TabaiInt(100) % TabaiInt(3)
print(g.to_cpu())
#>> 1

# divmod
q, r = divmod(TabaiInt(100), TabaiInt(7))
print(q.to_cpu(), r.to_cpu())
#>> 14 2
```

## テストの実行

```bash
pytest tabai/tests/test_operations.py
```

## ライセンス

MIT
