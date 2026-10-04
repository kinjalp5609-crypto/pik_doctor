"""Pack the selected PlantVillage images into one compact npz (72x72 RGB)."""
import os, sys, numpy as np, cv2, glob
SRC = sys.argv[1]            # .../PlantVillage-Dataset/raw/color
OUT = sys.argv[2]
S = int(sys.argv[3]) if len(sys.argv) > 3 else 72
classes = sorted(d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d)))
X, y, names = [], [], []
for ci, c in enumerate(classes):
    for f in sorted(glob.glob(os.path.join(SRC, c, '*'))):
        im = cv2.imread(f)
        if im is None: continue
        im = cv2.cvtColor(cv2.resize(im, (S, S), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        X.append(im); y.append(ci); names.append(os.path.basename(f))
np.savez_compressed(OUT, X=np.stack(X), y=np.array(y), classes=np.array(classes), names=np.array(names))
print(len(X), 'images', len(classes), 'classes')
