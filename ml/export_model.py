"""Export trained weights -> app/model/{weights.bin,manifest.json}; fit temperature on the validation split."""
import numpy as np, json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from cnn import Net, F32
RUN, DATA, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
d = np.load(DATA); X, y, classes = d['X'], d['y'], [str(c) for c in d['classes']]
sp = np.load(os.path.join(RUN, 'split.npz')); w = np.load(os.path.join(RUN, 'best.npz'))
ch = tuple(w[f'c{i}b'].shape[0] for i in range(4)); hid = w['f0b'].shape[0]
size = int(round((w['f0w'].shape[0] // ch[-1]) ** .5 * 2 ** len(ch)))
net = Net(len(classes), size=size, ch=ch, hid=hid)
for k in net.p: net.p[k] = w[k].astype(F32)
PAD = int(sys.argv[4]) if len(sys.argv) > 4 else 72
def center(a):
    o = (PAD - size)//2; return ((a[:, o:o+size, o:o+size, :].astype(F32)/255.) - .5)/.25
def logits(idx):
    return np.concatenate([net.forward(center(X[idx[i:i+256]]))[0] for i in range(0, len(idx), 256)])
Lv = logits(sp['va']); yv = y[sp['va']]
best_T, best_nll = 1, 1e9
for T in np.arange(.5, 4.01, .05):
    z = Lv/T; z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
    nll = -np.log(p[np.arange(len(yv)), yv] + 1e-9).mean()
    if nll < best_nll: best_T, best_nll = float(T), nll
print('temperature', best_T, 'val nll', best_nll)
os.makedirs(OUT, exist_ok=True)
tensors, off, blob = [], 0, bytearray()
for k in ['c0w','c0b','c1w','c1b','c2w','c2b','c3w','c3b','f0w','f0b','f1w','f1b']:
    a = np.ascontiguousarray(net.p[k], dtype='<f4'); b = a.tobytes()
    tensors.append({'name': k, 'shape': list(a.shape), 'offset': off, 'length': a.size}); blob += b; off += len(b)
open(os.path.join(OUT, 'weights.bin'), 'wb').write(blob)
json.dump({'classes': classes, 'size': size, 'pad': PAD, 'ch': list(ch), 'hid': hid, 'mean': .5, 'std': .25,
           'temperature': best_T, 'tensors': tensors, 'minConf': 0.60}, open(os.path.join(OUT, 'manifest.json'), 'w'), indent=1)
# test vectors for the JS engine (uint8 72x72 inputs + expected logits)
ti = sp['te'][:6]
np.savez(os.path.join(RUN, 'vectors.npz'), x=X[ti], logits=logits(ti), y=y[ti])
print('exported', len(blob)/1024, 'KB')
