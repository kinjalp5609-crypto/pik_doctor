"""OPTIONAL UPGRADE (not run in our build environment: no PyTorch / pretrained weights were reachable there).
Fine-tune an ImageNet-pretrained MobileNetV3-Small on the same 23 classes and export ONNX for a higher-accuracy model.
Run in Google Colab (free GPU):  !pip install timm onnx ; then run this file.
Dataset: https://huggingface.co/datasets/mohanty/PlantVillage  (or git clone github.com/spMohanty/PlantVillage-Dataset, use raw/color)
Keep only these crops: Tomato, Grape, Corn_(maize), Potato, Pepper,_bell.
NOTE: the web app's engine.js runs our plain CNN. To use a MobileNet, add onnxruntime-web (vendor it into app/ for offline use)."""
import os, torch, timm, glob, random
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms as T
from PIL import Image
ROOT = 'PlantVillage-Dataset/raw/color'; KEEP = ('Tomato', 'Grape', 'Corn', 'Potato', 'Pepper')
classes = sorted(d for d in os.listdir(ROOT) if d.startswith(KEEP))
files = [(f, i) for i, c in enumerate(classes) for f in glob.glob(f'{ROOT}/{c}/*')]
random.seed(42); random.shuffle(files); n = len(files); tr, va = files[:int(.8*n)], files[int(.8*n):int(.9*n)]
norm = T.Normalize([.485, .456, .406], [.229, .224, .225])
aug = T.Compose([T.RandomResizedCrop(224, (.6, 1)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.ColorJitter(.3, .3, .3, .05), T.ToTensor(), norm])
ev = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), norm])
class DS(Dataset):
    def __init__(s, items, tf): s.items, s.tf = items, tf
    def __len__(s): return len(s.items)
    def __getitem__(s, i): f, y = s.items[i]; return s.tf(Image.open(f).convert('RGB')), y
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
m = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=len(classes)).to(dev)
opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2); ce = nn.CrossEntropyLoss(label_smoothing=.05)
tl, vl = DataLoader(DS(tr, aug), 64, shuffle=True, num_workers=2), DataLoader(DS(va, ev), 128)
for ep in range(8):
    m.train()
    for x, y in tl: opt.zero_grad(); ce(m(x.to(dev)), y.to(dev)).backward(); opt.step()
    m.eval(); ok = sum((m(x.to(dev)).argmax(1).cpu() == y).sum().item() for x, y in vl); print(ep, 'val acc', ok/len(va))
torch.onnx.export(m.cpu().eval(), torch.randn(1, 3, 224, 224), 'pik_mobilenetv3.onnx', input_names=['x'], output_names=['logits'], opset_version=17)
open('classes.txt', 'w').write('\n'.join(classes))
