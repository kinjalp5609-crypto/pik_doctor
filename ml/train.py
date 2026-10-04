"""Train the Pik Doctor CNN (NumPy only) on packed PlantVillage subset."""
import numpy as np, json, time, sys, os
from cnn import Net, F32

DATA, OUT = sys.argv[1], sys.argv[2]
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
BS, LR = 64, 2e-3
SIZE, PAD = int(os.environ.get('SIZE',64)), int(os.environ.get('PAD',72))
CH = tuple(int(v) for v in os.environ.get('CH','16,32,64,64').split(','))
HID = int(os.environ.get('HID',128))
rng = np.random.default_rng(42)
d = np.load(DATA)
X, y, classes = d['X'], d['y'], [str(c) for c in d['classes']]
K = len(classes)

# stratified 80/10/10 split
tr, va, te = [], [], []
for k in range(K):
    idx = np.where(y == k)[0]; rng.shuffle(idx)
    a, b = int(.8*len(idx)), int(.9*len(idx))
    tr += list(idx[:a]); va += list(idx[a:b]); te += list(idx[b:])
tr, va, te = map(np.array, (tr, va, te))
np.savez(os.path.join(OUT, 'split.npz'), tr=tr, va=va, te=te)
print('train/val/test', len(tr), len(va), len(te), flush=True)

def norm(a): return ((a.astype(F32) / 255.0) - 0.5) / 0.25

def center(a):
    o = (PAD - SIZE)//2
    return norm(a[:, o:o+SIZE, o:o+SIZE, :])

def augment(a):
    n = len(a); out = np.empty((n, SIZE, SIZE, 3), F32)
    for i in range(n):
        oy, ox = rng.integers(0, PAD-SIZE+1, 2)
        im = a[i, oy:oy+SIZE, ox:ox+SIZE, :].astype(F32)
        if rng.random() < .5: im = im[:, ::-1]
        if rng.random() < .5: im = im[::-1]
        im = np.rot90(im, rng.integers(0, 4))
        gain = rng.uniform(.92, 1.08, 3).astype(F32)           # white-balance drift
        c = rng.uniform(.75, 1.25); br = rng.uniform(-25, 25)  # contrast / brightness
        im = (im - 128) * c + 128 + br
        im = im * gain
        out[i] = np.clip(im, 0, 255)
    return norm(out)

net = Net(K, seed=1, size=SIZE, ch=CH, hid=HID)
m = {k: np.zeros_like(v) for k, v in net.p.items()}
v = {k: np.zeros_like(p) for k, p in net.p.items()}
t = 0
def evaluate(idx):
    L = []
    for i in range(0, len(idx), 256):
        L.append(net.forward(center(X[idx[i:i+256]]))[0])
    L = np.concatenate(L); return L, (L.argmax(1) == y[idx]).mean()

best, steps = 0, EPOCHS * (len(tr)//BS)
for ep in range(EPOCHS):
    t0 = time.time(); perm = rng.permutation(tr); tl = 0
    for bi in range(len(tr)//BS):
        ids = perm[bi*BS:(bi+1)*BS]
        xb, yb = augment(X[ids]), y[ids]
        logits, st = net.forward(xb, True, 0.3, rng)
        z = logits - logits.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        tgt = np.full_like(p, .05/(K-1)); tgt[np.arange(len(yb)), yb] = .95   # label smoothing
        tl += -(tgt*np.log(p+1e-9)).sum(1).mean()
        g = net.backward(((p - tgt)/len(yb)).astype(F32), st)
        lr = LR * 0.5*(1+np.cos(np.pi*t/steps)) if t > 50 else LR*t/50
        t += 1
        for k in net.p:
            gk = g[k] + (1e-4*net.p[k] if k.endswith('w') else 0)
            m[k] = .9*m[k] + .1*gk; v[k] = .999*v[k] + .001*gk*gk
            net.p[k] -= (lr * (m[k]/(1-.9**t)) / (np.sqrt(v[k]/(1-.999**t)) + 1e-8)).astype(F32)
    _, va_acc = evaluate(va)
    print(f'ep {ep+1}/{EPOCHS} loss {tl/(len(tr)//BS):.3f} val_acc {va_acc:.4f} ({time.time()-t0:.0f}s)', flush=True)
    if va_acc >= best:
        best = va_acc; np.savez(os.path.join(OUT, 'best.npz'), **net.p)
print('best val', best)
