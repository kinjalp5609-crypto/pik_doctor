# Pik Doctor (पीक डॉक्टर)

**An offline crop-leaf disease checker for Maharashtra farmers, in Marathi, Hindi and English.**
Challenge 4: *Small AI for Development*, agriculture track.

A farmer photographs a leaf (or picks one from the gallery), and **the phone itself** says what the problem probably is, how urgent it is, what to do now and how to prevent it. It reads the answer aloud. After the first visit **it needs no internet, no server and no account**: the model, the advice and the app are all cached on the device.

- Crops covered: **tomato, grape (Nashik), maize, potato, bell pepper**, 23 leaf conditions (diseases + healthy)
- Model: a **0.75-million-parameter CNN (3 MB)** trained from scratch, executed in plain JavaScript, with no ML framework in the browser
- Runs on the devices people already have: any phone with a modern browser, installable as an app (PWA), about 0.4 s per diagnosis on a desktop-class CPU
- Safe by design: says **"not sure"** when confidence is low, says **"not a covered crop"** for other plants, never gives pesticide doses, always points to the local KVK and the Kisan Call Centre

## Results (held-out test images, never used for training or tuning)

| Measure | Result |
|---|---|
| Accuracy, 23 classes, crop not chosen | **94.7 %** (n = 994) |
| Accuracy when the farmer picks the crop | **96.2 %** |
| Accuracy on the answers it commits to (confidence ≥ 70 %, 89 % of photos) | **98.3 %** |
| Diseased leaf called "healthy" (crop not chosen) | 0.6 % (0.2 % with crop chosen) |
| Other-crop leaves correctly sent to "not covered / not sure" (600 held-out images) | 97.5 % |
| Model size / inference | 3 MB / ~0.4 s (desktop headless Chromium, with 4-way test-time augmentation) |

Full numbers per class: `docs/metrics.json`. Training log: `docs/train_log.txt`.

### Honest limitations (please read)
1. **Lab photos, not field photos.** PlantVillage images are single leaves on plain backgrounds. Accuracy on messy field photos will be lower. The app asks for one leaf on a plain background in daylight. A field-photo test set from Maharashtra is the most valuable next step.
2. **Near-duplicate images** exist in PlantVillage (several shots of the same leaf), so the test numbers are probably optimistic. The split is random by image, not by leaf.
3. **The "not covered" class is trained on other PlantVillage crops and synthetic images.** It has not seen real-world junk such as hands, soil or sky. The 97.5 % figure applies to other-crop leaves, not to arbitrary photos. We tried an unsupervised Mahalanobis-distance guard first; it flagged only 8 % of other-crop leaves, so we trained a real "other" class instead.
4. **Hindi and Marathi text was written by an AI model and has not been reviewed by a native-speaking agriculture extension worker.** Review `ml/build_kb.py` before any real-world use. The Kisan Call Centre number (1800-180-1551) should also be re-verified.
5. **Advice is general integrated-management guidance, not a prescription.** It deliberately names no pesticide, brand or dose; it sends the farmer to the KVK or agriculture officer.
6. Read-aloud uses the phone's own text-to-speech voices. If Marathi or Hindi voices are not installed, the app says so. Voice *input* is not included because offline speech recognition is not reliably available in browsers.

## How it works

```
photo ─► centre-crop ─► exact area-resize to 72 px ─► CNN (4 conv blocks + 2 dense, plain JS)
      ─► temperature-calibrated softmax over the chosen crop's classes (+ "not covered")
      ─► confidence ≥ 70 %?  yes: diagnosis + advice (MR / HI / EN) + read aloud
                             no : "not sure, retake the photo" + best guesses + KVK / helpline
```

- `ml/cnn.py`: forward and backward passes of the CNN written in NumPy (gradient-checked), because PyTorch, TensorFlow and pretrained weights could not be installed in our build environment.
- `ml/train.py`: augmentation (crops, flips, rotations, brightness / contrast / white-balance jitter), Adam, cosine schedule, label smoothing, stratified 80/10/10 split.
- `ml/export_model.py`: writes `weights.bin` + `manifest.json` and fits a **temperature** on the validation split so that confidence means something.
- `app/engine.js`: dependency-free JS inference engine. `tests/engine.test.js` proves it reproduces the NumPy logits (max difference 2e-6) and that its image resize matches OpenCV `INTER_AREA`.
- `app/sw.js`: service worker, cache-first: after the first load, the app, model, advice and sample photos all work in airplane mode (`tests/e2e_offline.py` checks this in headless Chromium with the network switched off).
- `ml/build_kb.py`: the trilingual knowledge base (`app/data/kb.json`), built from reusable advice snippets so each sentence is written and reviewed once.

## Run it

```bash
cd app && python3 -m http.server 8000     # then open http://localhost:8000
```

On a phone: open the hosted URL once with internet, then use "Install app" (or "Add to Home screen"). From then on it works offline.

### Deploy (Vercel)
`vercel.json` serves the `app/` folder as a static site: `vercel --prod` from the repo root, or import the repo in the Vercel dashboard. Nothing else to configure.

### Tests
```bash
node tests/engine.test.js app/model <vectors.json>   # JS engine == NumPy reference
python3 tests/e2e_offline.py                         # needs: pip install playwright; offline browser test
```

### Retrain
See `dataset/README.txt` for how to rebuild the dataset subset, then:
```bash
python3 ml/prep.py <PlantVillage>/raw/color data108.npz 108
python3 ml/prep_other.py data108.npz <other_crops_dir> data108o.npz 108
SIZE=96 PAD=108 CH=24,48,64,96 HID=192 python3 ml/train.py data108o.npz run 30
python3 ml/export_model.py run data108o.npz app/model 108   # then set "minConf": 0.7 in manifest.json
python3 ml/evaluate.py run data108o.npz app/model <held_out_other_crops> docs/metrics.json
```
Requirements: Python 3 with `numpy`, `opencv-python`; Node 18+ for the engine test; `playwright` for the e2e test. The app itself has **no dependencies**.

Optional accuracy upgrade: `ml/colab_train_mobilenet.py` fine-tunes a pretrained MobileNetV3 (untested here because pretrained weights were unreachable in our environment).

## What we would do with more time
Field photos from Maharashtra farms (cotton, soybean, sugarcane, onion, pomegranate, which PlantVillage does not cover), native-speaker review of every sentence, voice input with an on-device speech model, and a "send to KVK" button that queues a case for when the phone gets signal.

## Data and credits
Leaf images: PlantVillage (Hughes & Salathé 2015; Mohanty et al. 2016). Not redistributed here except six held-out demo photos in `app/samples` (provenance in `docs/samples_provenance.json`). Team credits: *add your names here*.

*Pik Doctor is a screening aid, not a diagnosis. Confirm with an agriculture expert before spraying anything.*


## Low-literacy design (v3)
- Camera-first home: one big camera button; crop tiles are optional; every screen can be heard (🔊) with auto-voice on by default.
- Result opens with a red / amber / green banner, then the condition, a 5-dot confidence ("Very sure / Fairly sure"), and 2-3 icon steps.
- One-tap actions: call the Kisan Call Centre (tel: link), "Show to shop" full-screen card (local language + English), WhatsApp share (needs internet), listen again.
- Photo coach: dark, glaring or blurry photos get a spoken retake hint before diagnosis ("use this photo anyway" is available).
- Tests: `python tests/e2e_ui.py` (offline reload, results, shop card, coach). Usability protocol: `docs/usability_test.md`.
- Needs review by a native Marathi/Hindi speaker: all new `ui` strings in `app/data/kb.json`. Verify that Marathi text-to-speech works offline on a real Android phone; if not, ship recorded audio.
