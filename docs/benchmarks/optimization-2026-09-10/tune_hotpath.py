import json
import statistics
import time
from pathlib import Path
import cupy as cp
from tabai_gpu import core
from tabai_gpu.core import GPUBigInt
from tabai_gpu.multi_gpu import _DistributedMultiplier
import importlib
multi = importlib.import_module('tabai_gpu.multi_gpu')
records=[]
engine=GPUBigInt()
for logn in [17, 24]:
    n=1 << logn
    data=cp.arange(n,dtype=cp.uint64)
    w_fwd,w_inv,_=engine._get_ntt_tables(n)
    cp.cuda.get_current_stream().synchronize()
    for tile in [128,256,512,1024]:
        core._NTT_TILE=tile
        for name,w in [('_ntt_forward',w_fwd),('_ntt_inverse',w_inv)]:
            fn=getattr(engine,name)
            for _ in range(2): fn(data,n,w)
            cp.cuda.get_current_stream().synchronize()
            samples=[]
            for _ in range(7):
                start,end=cp.cuda.Event(),cp.cuda.Event()
                start.record(); fn(data,n,w); end.record(); end.synchronize()
                samples.append(cp.cuda.get_elapsed_time(start,end))
            record={'kind':'ntt','logn':logn,'tile':tile,'direction':name,'median_ms':statistics.median(samples)}
            print(json.dumps(record),flush=True); records.append(record)
calc=_DistributedMultiplier((0,1),transfer='host')
try:
    for source_device,destination_device in [(0,1),(1,0)]:
        with cp.cuda.Device(source_device):
            source=cp.full(8*1024*1024,123,dtype=cp.uint64)
            cp.cuda.get_current_stream().synchronize()
        rank=calc.ranks[destination_device]
        with cp.cuda.Device(destination_device),rank.stream:
            target=cp.empty(source.size,dtype=source.dtype)
            for chunk in [1,2,4,8,16,128]:
                multi._HOST_CHUNK_BYTES=chunk*1024*1024
                samples=[]
                for iteration in range(9):
                    begin=time.perf_counter(); calc._copy(target,source,rank.stream); rank.stream.synchronize()
                    if iteration>=2: samples.append(1000*(time.perf_counter()-begin))
                assert bool(cp.all(target==123))
                record={'kind':'transfer','src':source_device,'dst':destination_device,'bytes':source.nbytes,'chunk_MiB':chunk,'median_ms':statistics.median(samples)}
                print(json.dumps(record),flush=True); records.append(record)
finally:
    calc.close()
Path('/mnt/performance-tuning.json').write_text(json.dumps(records,indent=2)+'\n')
