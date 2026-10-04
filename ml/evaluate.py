"""Honest evaluation on the held-out test split + out-of-scope crops (rejection test)."""
import numpy as np, json, sys, os, glob, cv2
sys.path.insert(0, os.path.dirname(__file__))
from cnn import Net, F32
RUN, DATA, MODEL, OODDIR, OUTJSON = sys.argv[1:6]
man = json.load(open(os.path.join(MODEL, 'manifest.json')))
classes, size, PAD, T = man['classes'], man['size'], man['pad'], man['temperature']
d = np.load(DATA); X, y = d['X'], d['y']; sp = np.load(os.path.join(RUN, 'split.npz')); w = np.load(os.path.join(RUN, 'best.npz'))
net = Net(len(classes), size=size, ch=tuple(man['ch']), hid=man['hid'])
for k in net.p: net.p[k] = w[k].astype(F32)
def center(a):
    o = (PAD-size)//2; return ((a[:, o:o+size, o:o+size, :].astype(F32)/255.)-.5)/.25
def probs(A, mask=None, tta=True):
    # same as the app: average class probabilities over identity / h-flip / v-flip / both (TTA)
    outs = []
    for f in ([lambda a: a, lambda a: a[:, :, ::-1], lambda a: a[:, ::-1], lambda a: a[:, ::-1, ::-1]] if tta else [lambda a: a]):
        L = np.concatenate([net.forward(np.ascontiguousarray(f(center(A[i:i+256]))))[0] for i in range(0, len(A), 256)]) / T
        if mask is not None: L = np.where(mask, L, -1e9)
        L -= L.max(1, keepdims=True); p = np.exp(L); outs.append(p/p.sum(1, keepdims=True))
    return np.mean(outs, 0)
te = sp['te']; yt = y[te]; P = probs(X[te]); pred = P.argmax(1)
crop = np.array([c.split('___')[0] for c in classes])
M = (crop[None, :] == crop[yt][:, None]) | (crop[None, :] == 'Other')
Pc = probs(X[te], M); predc = Pc.argmax(1)   # crop known, but 'not covered' stays possible
P0 = probs(X[te], tta=False)
res = {'n_test': int(len(te)), 'acc_auto_no_tta': float((P0.argmax(1) == yt).mean()), 'acc_auto': float((pred == yt).mean()), 'acc_crop_known': float((predc == yt).mean())}
per = {}
for k, c in enumerate(classes):
    m = yt == k; tp = ((pred == k) & m).sum(); fp = ((pred == k) & ~m).sum()
    per[c] = {'n': int(m.sum()), 'recall': float(tp/max(m.sum(), 1)), 'precision': float(tp/max(tp+fp, 1))}
res['per_class'] = per
# healthy-vs-diseased safety: how often a diseased leaf is called healthy
hid = np.array([c.endswith('healthy') for c in classes]); OT = classes.index('Other___not_covered') if 'Other___not_covered' in classes else -1
dis = ~hid[yt]; res['diseased_called_healthy_auto'] = float(hid[pred][dis].mean()); res['diseased_called_healthy_crop_known'] = float(hid[predc][dis].mean())
# selective prediction
sel = {}
for thr in [.5, .6, .7, .8, .9]:
    for name, PP, pr in [('auto', P, pred), ('crop_known', Pc, predc)]:
        keep = PP.max(1) >= thr
        sel[f'{name}@{thr}'] = {'coverage': float(keep.mean()), 'acc_on_answered': float((pr[keep] == yt[keep]).mean())}
res['selective'] = sel
# out-of-scope crops
ood = []
for f in sorted(glob.glob(os.path.join(OODDIR, '*', '*'))):
    im = cv2.imread(f)
    if im is not None: ood.append(cv2.cvtColor(cv2.resize(im, (PAD, PAD), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
if ood:
    Pp = probs(np.stack(ood)); Po = Pp.max(1); top = Pp.argmax(1)
    res['ood_n'] = len(ood)
    res['ood_not_covered_or_unsure'] = {str(t): float(((top == OT) | (Po < t)).mean()) for t in [.5, .6, .7, .8, .9]}
    res['ood_labelled_as_a_covered_disease_confidently_at_0.7'] = float(((top != OT) & (Po >= .7)).mean())
json.dump(res, open(OUTJSON, 'w'), indent=1)
print('acc auto %.4f | crop known %.4f | diseased->healthy auto %.4f crop %.4f' % (res['acc_auto'], res['acc_crop_known'], res['diseased_called_healthy_auto'], res['diseased_called_healthy_crop_known']))
for k, v in sel.items(): print(k, v)
print('OOD not-covered-or-unsure', res.get('ood_not_covered_or_unsure'), '| confidently wrong @0.7', res.get('ood_labelled_as_a_covered_disease_confidently_at_0.7'))
worst = sorted(per.items(), key=lambda kv: kv[1]['recall'])[:6]
print('weakest', [(k, round(v['recall'], 3)) for k, v in worst])
