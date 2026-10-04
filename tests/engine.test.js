// Verifies the JS engine against NumPy reference outputs. Usage: node tests/engine.test.js <model_dir> <vectors.json>
const fs = require('fs'), path = require('path'), E = require('../app/engine.js');
const dir = process.argv[2], vec = JSON.parse(fs.readFileSync(process.argv[3]));
const m = JSON.parse(fs.readFileSync(path.join(dir, 'manifest.json')));
const b = fs.readFileSync(path.join(dir, 'weights.bin')); const buf = b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
const model = E.buildModel(m, buf);
let maxErr = 0, agree = 0;
vec.x.forEach((b64, k) => {
  const u8 = Buffer.from(b64, 'base64'), n = Math.round(Math.sqrt(u8.length / 3)), rgba = new Uint8ClampedArray(n * n * 4);
  for (let i = 0; i < n * n; i++) { rgba[i*4] = u8[i*3]; rgba[i*4+1] = u8[i*3+1]; rgba[i*4+2] = u8[i*3+2]; rgba[i*4+3] = 255; }
  const x = E.preprocess(rgba, n, n, Object.assign({}, m, { pad: n }));
  const z = E.logits(model, x);
  z.forEach((v, i) => maxErr = Math.max(maxErr, Math.abs(v - vec.logits[k][i])));
  agree += z.indexOf(Math.max(...z)) === vec.logits[k].indexOf(Math.max(...vec.logits[k])) ? 1 : 0;
});
console.log('engine vs numpy: max |logit diff| =', maxErr.toExponential(2), '| argmax agree', agree + '/' + vec.x.length);
// preprocessing check: JS area-resize of the full image vs OpenCV INTER_AREA
const f = vec.full, rgba = new Uint8ClampedArray(Buffer.from(f.rgba, 'base64')), ref = Buffer.from(f.small72, 'base64');
const out = E.areaResize(rgba, f.w, f.h, 72); let d = 0, mx = 0;
for (let i = 0; i < out.length; i++) { const e = Math.abs(Math.round(out[i]) - ref[i]); d += e; mx = Math.max(mx, e); }
console.log('resize vs OpenCV INTER_AREA: mean abs diff', (d / out.length).toFixed(3), 'max', mx);
if (maxErr > 1e-3 || agree !== vec.x.length || mx > 3) { console.error('FAIL'); process.exit(1); }
console.log('PASS');
