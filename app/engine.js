/* Pik Doctor on-device inference engine: no dependencies, runs the exported CNN in plain JS.
   Works in the browser (window.PikEngine) and in Node (module.exports) so it can be unit-tested. */
(function (root) {
  'use strict';

  async function loadModel(base) {
    const m = await (await fetch(base + 'manifest.json')).json();
    const buf = await (await fetch(base + 'weights.bin')).arrayBuffer();
    return buildModel(m, buf);
  }

  function buildModel(m, buf) {
    const T = {};
    for (const t of m.tensors) T[t.name] = new Float32Array(buf, t.offset, t.length);
    return { m, T };
  }

  // 3x3 conv, pad 1, NHWC, followed by ReLU and 2x2 max-pool (fused to save memory)
  function convReluPool(x, H, W, C, w, b, O) {
    const out = new Float32Array(H * W * O);
    for (let y = 0; y < H; y++) {
      for (let xx = 0; xx < W; xx++) {
        const ob = (y * W + xx) * O;
        for (let o = 0; o < O; o++) out[ob + o] = b[o];
        for (let ky = 0; ky < 3; ky++) {
          const iy = y + ky - 1; if (iy < 0 || iy >= H) continue;
          for (let kx = 0; kx < 3; kx++) {
            const ix = xx + kx - 1; if (ix < 0 || ix >= W) continue;
            const ib = (iy * W + ix) * C;
            const wb = ((ky * 3 + kx) * C) * O;
            for (let c = 0; c < C; c++) {
              const v = x[ib + c]; if (v === 0) continue;
              const wo = wb + c * O;
              for (let o = 0; o < O; o++) out[ob + o] += v * w[wo + o];
            }
          }
        }
        for (let o = 0; o < O; o++) if (out[ob + o] < 0) out[ob + o] = 0;
      }
    }
    const h2 = H >> 1, w2 = W >> 1, pooled = new Float32Array(h2 * w2 * O);
    for (let y = 0; y < h2; y++) for (let xx = 0; xx < w2; xx++) for (let o = 0; o < O; o++) {
      const a = out[((2 * y) * W + 2 * xx) * O + o], b2 = out[((2 * y) * W + 2 * xx + 1) * O + o];
      const c2 = out[((2 * y + 1) * W + 2 * xx) * O + o], d = out[((2 * y + 1) * W + 2 * xx + 1) * O + o];
      pooled[(y * w2 + xx) * O + o] = Math.max(a, b2, c2, d);
    }
    return pooled;
  }

  function dense(x, w, b, nIn, nOut, relu) {
    const out = new Float32Array(nOut);
    for (let o = 0; o < nOut; o++) out[o] = b[o];
    for (let i = 0; i < nIn; i++) {
      const v = x[i]; if (v === 0) continue;
      const wo = i * nOut;
      for (let o = 0; o < nOut; o++) out[o] += v * w[wo + o];
    }
    if (relu) for (let o = 0; o < nOut; o++) if (out[o] < 0) out[o] = 0;
    return out;
  }

  // input: Float32Array size*size*3, already normalised
  function logits(model, input) {
    const { m, T } = model;
    let x = input, s = m.size, c = 3;
    for (let i = 0; i < m.ch.length; i++) {
      x = convReluPool(x, s, s, c, T['c' + i + 'w'], T['c' + i + 'b'], m.ch[i]);
      s >>= 1; c = m.ch[i];
    }
    const h = dense(x, T.f0w, T.f0b, x.length, m.hid, true);
    return dense(h, T.f1w, T.f1b, m.hid, m.classes.length, false);
  }

  // Fractional-area downscale (same maths as OpenCV INTER_AREA) from RGBA w*h to n*n, returns RGB float 0..255
  function areaResize(rgba, w, h, n) {
    const wx = axisWeights(w, n), wy = axisWeights(h, n);
    const tmp = new Float32Array(h * n * 3);
    for (let y = 0; y < h; y++) for (let dx = 0; dx < n; dx++) {
      const { start, ws } = wx[dx]; let r = 0, g = 0, b = 0;
      for (let k = 0; k < ws.length; k++) {
        const p = (y * w + start + k) * 4; const q = ws[k];
        r += rgba[p] * q; g += rgba[p + 1] * q; b += rgba[p + 2] * q;
      }
      const o = (y * n + dx) * 3; tmp[o] = r; tmp[o + 1] = g; tmp[o + 2] = b;
    }
    const out = new Float32Array(n * n * 3);
    for (let dy = 0; dy < n; dy++) for (let x = 0; x < n; x++) {
      const { start, ws } = wy[dy]; let r = 0, g = 0, b = 0;
      for (let k = 0; k < ws.length; k++) {
        const p = ((start + k) * n + x) * 3; const q = ws[k];
        r += tmp[p] * q; g += tmp[p + 1] * q; b += tmp[p + 2] * q;
      }
      const o = (dy * n + x) * 3; out[o] = r; out[o + 1] = g; out[o + 2] = b;
    }
    return out;
  }
  function axisWeights(src, n) {
    const scale = src / n, res = [];
    for (let d = 0; d < n; d++) {
      const a = d * scale, b = (d + 1) * scale;
      const i0 = Math.floor(a), i1 = Math.min(src - 1, Math.ceil(b) - 1), ws = [];
      for (let i = i0; i <= i1; i++) ws.push((Math.min(b, i + 1) - Math.max(a, i)) / scale);
      res.push({ start: i0, ws });
    }
    return res;
  }

  // RGBA of a square-ish image -> normalised 64x64x3 (resize to 72, centre crop 64)
  function preprocess(rgba, w, h, m) {
    const pad = m.pad || 72, size = m.size;
    const rgb = areaResize(rgba, w, h, pad), off = (pad - size) >> 1;
    const out = new Float32Array(size * size * 3);
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) for (let c = 0; c < 3; c++) {
      const v = Math.round(rgb[((y + off) * pad + (x + off)) * 3 + c]);   // match uint8 storage in training
      out[(y * size + x) * 3 + c] = ((v / 255) - m.mean) / m.std;
    }
    return out;
  }

  function flip(x, s, horiz) {   // NHWC square image flip
    const o = new Float32Array(x.length);
    for (let y = 0; y < s; y++) for (let xx = 0; xx < s; xx++) {
      const sy = horiz ? y : s - 1 - y, sx = horiz ? s - 1 - xx : xx;
      for (let c = 0; c < 3; c++) o[(y * s + xx) * 3 + c] = x[(sy * s + sx) * 3 + c];
    }
    return o;
  }

  // Test-time augmentation: average class probabilities over identity / h-flip / v-flip / both
  function classifyTTA(model, input, allowed) {
    const s = model.m.size, h = flip(input, s, true), v = flip(input, s, false), hv = flip(h, s, false);
    const runs = [input, h, v, hv].map(x => classify(model, x, allowed));
    const acc = new Map();
    runs.forEach(r => r.forEach(o => acc.set(o.i, (acc.get(o.i) || 0) + o.p / runs.length)));
    return [...acc.entries()].map(([i, p]) => ({ i, p })).sort((a, b) => b.p - a.p);
  }

  // classes: optional array of allowed class indices (crop filter). Returns sorted [{i,p}]
  function classify(model, input, allowed) {
    const m = model.m, z = logits(model, input);
    const idx = allowed && allowed.length ? allowed : Array.from(z, (_, i) => i);
    let mx = -Infinity; for (const i of idx) mx = Math.max(mx, z[i] / m.temperature);
    let sum = 0; const ps = idx.map(i => { const e = Math.exp(z[i] / m.temperature - mx); sum += e; return { i, p: e }; });
    ps.forEach(o => o.p /= sum);
    ps.sort((a, b) => b.p - a.p);
    return ps;
  }

  const api = { loadModel, buildModel, logits, preprocess, classify, classifyTTA, areaResize };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.PikEngine = api;
})(typeof self !== 'undefined' ? self : this);
