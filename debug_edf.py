from synthforge.simulator.ecg import write_edf, simulate_ecg_cohort
from synthforge.simulator.common import rng
import numpy as np, tempfile, pathlib
root=pathlib.Path(tempfile.mkdtemp())
sig,lab,_,leads=simulate_ecg_cohort(n=1, seed=1, fs=250)
p=write_edf(root/'x.edf', sig[0], 250, leads)
raw=p.read_bytes()
print('len', len(raw), 'headerbytes field', raw[184:192])
print('ns field', raw[252:256])
print('first 256', raw[:256])
print('signal header first 512', raw[256:512])
pos=256
for k,w in [('label',16),('transducer',80),('dimension',8),('pmin',8),('pmax',8),('dmin',8),('dmax',8),('prefilter',80),('spr',8),('reserved',32)]:
    print(k, [raw[pos+i*w:pos+(i+1)*w] for i in range(3)])
    pos+=w*12
print('pos after', pos)
