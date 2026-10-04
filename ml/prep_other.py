"""Append an 'Other___not_covered' class: other-crop leaves (PlantVillage crops we do NOT cover) + synthetic non-leaf images.
This teaches the model to say 'not a covered crop' instead of forcing a wrong tomato/grape/... label."""
import numpy as np, cv2, glob, os, sys
SRC_NPZ, OOD_TRAIN, OUT, S = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
d = np.load(SRC_NPZ); X, y, classes, names = d['X'], d['y'], list(d['classes']), list(d['names'])
rng = np.random.default_rng(5); imgs = []; nm = []
for f in sorted(glob.glob(os.path.join(OOD_TRAIN, '*', '*'))):
    im = cv2.imread(f)
    if im is not None: imgs.append(cv2.cvtColor(cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)); nm.append('real_' + os.path.basename(f))
n_real = len(imgs)
def rc(): return tuple(int(v) for v in rng.integers(0, 256, 3))
for i in range(250):
    k = i % 5; im = np.zeros((S, S, 3), np.uint8)
    if k == 0: im[:] = rc()
    elif k == 1: im = np.repeat(np.linspace(rng.integers(0, 256, 3), rng.integers(0, 256, 3), S).astype(np.uint8)[None], S, 0)
    elif k == 2: im = rng.integers(0, 256, (S, S, 3), dtype=np.uint8)
    elif k == 3: im = cv2.GaussianBlur(rng.integers(0, 256, (S, S, 3), dtype=np.uint8), (0, 0), float(rng.uniform(1, 6)))
    else:   # UI / clutter: flat background with random shapes and text
        im[:] = rc()
        for _ in range(int(rng.integers(3, 9))):
            if rng.random() < .5: cv2.rectangle(im, tuple(int(v) for v in rng.integers(0, S, 2)), tuple(int(v) for v in rng.integers(0, S, 2)), rc(), -1)
            else: cv2.putText(im, 'Ab', tuple(int(v) for v in rng.integers(0, S-10, 2)), cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(.4, 1.5)), rc(), 2)
    if rng.random() < .5: im = cv2.GaussianBlur(im, (0, 0), float(rng.uniform(.3, 1.5)))
    imgs.append(im); nm.append(f'synthetic_{i}')
K = len(classes)
np.savez_compressed(OUT, X=np.concatenate([X, np.stack(imgs)]), y=np.concatenate([y, np.full(len(imgs), K)]),
                    classes=np.array(classes + ['Other___not_covered']), names=np.array(names + nm))
print('other class:', n_real, 'real +', len(imgs) - n_real, 'synthetic; total', len(X) + len(imgs))
