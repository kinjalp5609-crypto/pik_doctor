"""Tiny CNN in pure NumPy (forward + backward). NHWC layout, float32.
Kept dependency-free on purpose: the same weights are run in the browser by app/engine.js."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

F32 = np.float32

def conv_fwd(x, W, b):
    # x: N,H,W,C   W: 3,3,C,O  (pad 1, stride 1)
    N, H, Wd, C = x.shape
    O = W.shape[-1]
    xp = np.pad(x, ((0,0),(1,1),(1,1),(0,0)))
    win = swv(xp, (3,3), axis=(1,2))                 # N,H,W,C,3,3
    col = np.ascontiguousarray(win.transpose(0,1,2,4,5,3)).reshape(N*H*Wd, 9*C)
    out = col @ W.reshape(9*C, O) + b
    return out.reshape(N, H, Wd, O), (col, x.shape)

def conv_bwd(dout, W, cache):
    col, xs = cache
    N, H, Wd, C = xs
    O = W.shape[-1]
    d = dout.reshape(N*H*Wd, O)
    dW = (col.T @ d).reshape(3,3,C,O)
    db = d.sum(0)
    dcol = (d @ W.reshape(9*C, O).T).reshape(N, H, Wd, 3, 3, C)
    dxp = np.zeros((N, H+2, Wd+2, C), F32)
    for ky in range(3):
        for kx in range(3):
            dxp[:, ky:ky+H, kx:kx+Wd, :] += dcol[:, :, :, ky, kx, :]
    return dxp[:, 1:-1, 1:-1, :], dW, db

def pool_fwd(x):
    N, H, W, C = x.shape
    r = x.reshape(N, H//2, 2, W//2, 2, C)
    out = r.max(axis=(2,4))
    return out, (r, out)

def pool_bwd(dout, cache):
    r, out = cache
    mask = (r == out[:, :, None, :, None, :]).astype(F32)
    mask /= mask.sum(axis=(2,4), keepdims=True)
    dr = mask * dout[:, :, None, :, None, :]
    N, h, _, w, _, C = r.shape
    return dr.reshape(N, h*2, w*2, C)

class Net:
    # 64x64x3 -> conv16 -> pool -> conv32 -> pool -> conv64 -> pool -> conv64 -> pool -> fc128 -> fc K
    def __init__(self, K, seed=0, size=64, ch=(16,32,64,64), hid=128):
        rng = np.random.default_rng(seed)
        self.size, self.ch, self.hid, self.K = size, ch, hid, K
        self.p = {}
        cin = 3
        for i, c in enumerate(ch):
            self.p[f'c{i}w'] = (rng.standard_normal((3,3,cin,c)) * np.sqrt(2/(9*cin))).astype(F32)
            self.p[f'c{i}b'] = np.zeros(c, F32)
            cin = c
        flat = (size // 2**len(ch))**2 * ch[-1]
        self.p['f0w'] = (rng.standard_normal((flat, hid)) * np.sqrt(2/flat)).astype(F32)
        self.p['f0b'] = np.zeros(hid, F32)
        self.p['f1w'] = (rng.standard_normal((hid, K)) * np.sqrt(1/hid)).astype(F32)
        self.p['f1b'] = np.zeros(K, F32)

    def forward(self, x, train=False, drop=0.3, rng=None):
        caches = []
        h = x
        for i in range(len(self.ch)):
            z, cc = conv_fwd(h, self.p[f'c{i}w'], self.p[f'c{i}b'])
            a = np.maximum(z, 0)
            h, pc = pool_fwd(a)
            caches.append((cc, z, pc))
        flat = h.reshape(h.shape[0], -1)
        z0 = flat @ self.p['f0w'] + self.p['f0b']
        a0 = np.maximum(z0, 0)
        if train and drop > 0:
            m = (rng.random(a0.shape) > drop).astype(F32) / F32(1 - drop)
            a0 = a0 * m
        else:
            m = None
        logits = a0 @ self.p['f1w'] + self.p['f1b']
        return logits, (caches, flat, h.shape, z0, a0, m)

    def backward(self, dlogits, st):
        caches, flat, hs, z0, a0, m = st
        g = {}
        g['f1w'] = a0.T @ dlogits
        g['f1b'] = dlogits.sum(0)
        da0 = dlogits @ self.p['f1w'].T
        if m is not None: da0 = da0 * m
        dz0 = da0 * (z0 > 0)
        g['f0w'] = flat.T @ dz0
        g['f0b'] = dz0.sum(0)
        dh = (dz0 @ self.p['f0w'].T).reshape(hs)
        for i in reversed(range(len(self.ch))):
            cc, z, pc = caches[i]
            da = pool_bwd(dh, pc)
            dz = da * (z > 0)
            dh, g[f'c{i}w'], g[f'c{i}b'] = conv_bwd(dz, self.p[f'c{i}w'], cc)
        return g
