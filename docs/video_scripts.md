# Video scripts (each max 60 s). Record with OBS/Loom on a phone or Chrome DevTools device mode.

## Demo video (60 s)
**0–10 s (face or caption):** "Farmers often spot a sick leaf where there is no signal. Pik Doctor diagnoses it on the phone itself, in Marathi, Hindi and English."
**10–20 s:** Show the app. Turn on **airplane mode** and say it. Show the "Offline: fully working" pill.
**20–40 s:** Pick *Tomato*, tap a sample (or take a real photo of a leaf). Result: name, "Act today" badge, confidence, what to do. Switch to **मराठी**, then **हिंदी**. Tap **ऐका** (read aloud).
**40–52 s:** Show a non-crop photo (an apple leaf or your hand): the app says "not a covered crop" and a blurry photo says "not sure". Say: "It would rather say 'I don't know' than guess wrong."
**52–60 s:** "94.7% accurate on held-out images, 3 MB, works with no internet. Pik Doctor."

## Tech video (60 s)
**0–15 s, stack:** "No ML framework in the browser. We trained a small CNN in pure NumPy with hand-written backprop, gradient-checked, because PyTorch and pretrained weights were not available in our environment. It runs in plain JavaScript. PWA with a service worker for offline."
**15–35 s, highlights:** "Calibrated confidence with temperature scaling, test-time augmentation, a 'not covered' class from other crops plus synthetic junk. The JS engine matches NumPy to 2e-6 and our resize matches OpenCV. Knowledge base: 29 reusable advice snippets in 3 languages."
**35–55 s, what failed:** "Our first model was only 88% and weak on tomato blight, so we went to 96-pixel inputs and reached 94.7%. Our first safety check failed: an apple leaf was called a tomato disease at 90%. A Mahalanobis out-of-distribution guard caught only 8%, so we trained a real 'other' class, which now refuses 97.5% of other-crop leaves."
**55–60 s:** "Lesson: calibration and refusal matter more than raw accuracy."

## Team video (60 s)
Placeholder: who you are, where you study or work, why Maharashtra agriculture, who did what. Show the team picture.
