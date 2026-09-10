import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
import statistics
import sys
import time
from pathlib import Path
import cupy as cp
sys.path.insert(0, '/mnt')
from benchmarks.common import make_inputs
from tabai_gpu import TabaiInt, multi_gpu
from tabai_gpu.core import GPUBigInt
from tabai_gpu.multi_gpu import _DistributedMultiplier

parser=argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args=parser.parse_args()
records=[]
for mode in ['single','multi']:
    with multi_gpu([0,1]) if mode=='multi' else nullcontext() as executor:
        for op in ['mul','square']:
            raw=make_inputs({'operation':op,'bits':100000000,'exponent':None},42)
            values=[TabaiInt(value) for value in raw]
            def run():
                return values[0]* (values[0] if op=='square' else values[1])
            def sync():
                if executor: executor.synchronize()
                cp.cuda.get_current_stream().synchronize()
            for _ in range(3):
                result=run(); sync(); del result
            events=[]; copies=[]; originals={}
            for name in ['_ntt_forward','_ntt_inverse','_resolve_carries','_recombine_chunks','_trim_z']:
                old=getattr(GPUBigInt,name); originals[name]=old
                def wrap(self,*a,_old=old,_name=name,**kw):
                    device=cp.cuda.runtime.getDevice()
                    start,end=cp.cuda.Event(),cp.cuda.Event()
                    start.record()
                    result=_old(self,*a,**kw)
                    end.record()
                    events.append((device,_name,start,end))
                    return result
                setattr(GPUBigInt,name,wrap)
            original_copy=_DistributedMultiplier._copy
            def copy(self,dst,src,stream):
                start=time.perf_counter()
                result=original_copy(self,dst,src,stream)
                copies.append({'src':src.device.id,'dst':dst.device.id,'bytes':src.nbytes,'wall_ms':1000*(time.perf_counter()-start)})
                return result
            _DistributedMultiplier._copy=copy
            times=[]
            try:
                for _ in range(5):
                    sync(); start=time.perf_counter(); result=run(); sync()
                    times.append(1000*(time.perf_counter()-start)); del result
            finally:
                for name,old in originals.items(): setattr(GPUBigInt,name,old)
                _DistributedMultiplier._copy=original_copy
            grouped=defaultdict(float)
            for device,name,start,end in events:
                with cp.cuda.Device(device):
                    grouped[f'device{device}:{name}']+=cp.cuda.get_elapsed_time(start,end)/5
            record={'mode':mode,'operation':op,'bits':100000000,'wall_median_ms':statistics.median(times),'gpu_event_ms_per_operation':dict(grouped),'copy_wall_ms_per_operation':sum(v['wall_ms'] for v in copies)/5,'copy_bytes_per_operation':sum(v['bytes'] for v in copies)/5,'note':'Instrumented warm-cache run; GPU events on different devices overlap and must not be summed as wall time.'}
            records.append(record)
            print(json.dumps(record),flush=True)
Path(args.output).write_text(json.dumps(records,indent=2)+'\n')
