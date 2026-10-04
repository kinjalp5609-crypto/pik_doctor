DATASET: PlantVillage (Hughes & Salathe 2015; Mohanty et al. 2016), subset used by Pik Doctor
Source: https://github.com/spMohanty/PlantVillage-Dataset (raw/color) and https://huggingface.co/datasets/mohanty/PlantVillage
We do NOT redistribute the images. To rebuild our subset:
  1. git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset
  2. check out raw/color/ for the 23 classes in app/model/manifest.json (plus an "Other___not_covered" class built from other PlantVillage crops and synthetic non-leaf images, see ml/prep_other.py) (we sampled up to 400 images per class, seed 0; see split_info.txt)
  3. python ml/prep.py <path>/raw/color data72.npz 72      # 72x72 RGB, INTER_AREA
  4. python ml/train.py data72.npz run 25
Columns / structure: one folder per class, "<Crop>___<Condition>", JPEG leaf photos on a plain lab background.
Held-out samples shipped in app/samples are listed in docs/samples_provenance.json.
