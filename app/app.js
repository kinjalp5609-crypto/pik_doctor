(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const CROP_EMOJI = { tomato: '🍅', grape: '🍇', corn: '🌽', potato: '🥔', pepper: '🫑' };
  const VOICE = { mr: 'mr-IN', hi: 'hi-IN', en: 'en-IN' };
  const SAMPLES = ['s1', 's2', 's3', 's4', 's5', 's6'];
  const KCC = '18001801551';
  // one picture per advice snippet, so a farmer can follow the steps without reading
  const ICON = {
    do_remove_leaves: '✂️', do_destroy_plants: '🗑️', do_water_base: '💧', do_no_wet_work: '🚫', do_expert: '📞', do_expert_urgent: '🚨',
    do_ventilate: '🌬️', do_whitefly: '🪰', do_wash_tools: '🧼', do_check_underside: '🔍', do_mites_water: '🚿', do_remove_mummies: '🗑️',
    do_mark_vines: '📍', do_monitor: '👀', do_healthy: '✅', do_rust_mild: '👀', do_potato_tubers: '🥔',
    pv_rotate: '🔄', pv_seed: '🌱', pv_spacing: '↔️', pv_residue: '🧹', pv_resistant: '🛡️', pv_nutrition: '🌾',
    pv_weeds: '🌿', pv_mulch: '🍂', pv_scout: '👀', pv_canopy: '✂️', pv_wound: '🩹', pv_nursery: '🏡'
  };
  const BLUR_MIN = 6; // Laplacian variance on the 256px view; values below this are clearly out of focus
  const st = { lang: 'en', crop: 'auto', kb: null, model: null, byId: {}, last: null, installEvt: null, auto: true, pending: null };

  function lsGet(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch (e) { return d; } }
  function lsSet(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage may be blocked */ } }
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const tr = (o) => (o && (o[st.lang] || o.en)) || '';
  const en = (o) => (o && o.en) || '';
  const U = (k) => tr(st.kb.ui[k]);

  function detectLang() {
    const saved = lsGet('pik.lang', '');
    if (['mr', 'hi', 'en'].includes(saved)) return saved;
    const n = (navigator.language || 'en').toLowerCase();
    return n.startsWith('hi') ? 'hi' : n.startsWith('en') ? 'en' : 'mr';
  }

  // ---------- boot ----------
  async function boot() {
    st.lang = detectLang(); st.auto = lsGet('pik.auto', '1') === '1';
    try {
      st.kb = await (await fetch('data/kb.json')).json();
      st.kb.classes.forEach((c) => { st.byId[c.id] = c; });
    } catch (e) { document.body.textContent = 'Could not load app data.'; return; }
    renderChrome(); updateNet();
    window.addEventListener('online', updateNet); window.addEventListener('offline', updateNet);
    $('modelPill').textContent = U('model_loading');
    try { st.model = await PikEngine.loadModel('model/'); $('modelPill').textContent = '✓ ' + U('model_ready'); }
    catch (e) { $('modelPill').textContent = '⚠ model'; }
    if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
  }

  // ---------- chrome / i18n ----------
  function renderChrome() {
    document.documentElement.lang = st.lang;
    $('appName').textContent = U('app_name'); $('tagline').textContent = U('tagline');
    document.querySelectorAll('[data-lang]').forEach((b) => {
      b.setAttribute('aria-pressed', String(b.dataset.lang === st.lang));
      b.onclick = () => { stopSpeak(); st.lang = b.dataset.lang; lsSet('pik.lang', st.lang); renderChrome(); updateNet(); if (st.model) $('modelPill').textContent = '✓ ' + U('model_ready'); if (!$('result').hidden && st.last) renderResult(st.last, false); };
    });
    $('homeLead').textContent = U('home_lead');
    $('howtoBtn').textContent = '🔊 ' + U('howto_btn'); $('howtoBtn').onclick = () => speak(U('howto_voice'));
    $('whichCrop').textContent = U('crop_optional');
    const box = $('crops'); box.innerHTML = '';
    const items = [['auto', '🔍', U('crop_auto')]].concat(Object.keys(st.kb.crops).map((k) => [k, CROP_EMOJI[k], tr(st.kb.crops[k])]));
    items.forEach(([k, e, label]) => {
      const b = document.createElement('button'); b.className = 'crop'; b.setAttribute('role', 'radio');
      b.setAttribute('aria-checked', String(st.crop === k)); b.innerHTML = '<span class="e">' + e + '</span><span>' + esc(label) + '</span>';
      b.onclick = () => { st.crop = k; renderChrome(); }; box.appendChild(b);
    });
    $('takeTxt').textContent = U('take_photo'); $('pickTxt').textContent = U('choose_photo');
    $('photoTip').textContent = U('photo_tip'); $('covered').textContent = U('covered');
    $('disclaimer').textContent = U('disclaimer'); $('busyTxt').textContent = U('checking');
    $('histTitle').textContent = U('history'); $('clearHist').textContent = U('clear');
    $('samplesTitle').textContent = ({ en: 'No leaf with you? Try a sample photo', hi: 'पत्ती पास नहीं है? नमूना फोटो आज़माएँ', mr: 'पान जवळ नाही? नमुना फोटो वापरून पहा' })[st.lang];
    $('installBtn').textContent = U('install');
    const vb = $('voiceBtn'); vb.textContent = (st.auto ? '🔊 ' : '🔇 ') + U('auto_voice'); vb.setAttribute('aria-pressed', String(st.auto));
    vb.onclick = () => { st.auto = !st.auto; lsSet('pik.auto', st.auto ? '1' : '0'); if (!st.auto) stopSpeak(); renderChrome(); };
    renderSamples(); renderHistory();
  }
  function updateNet() {
    const on = navigator.onLine, p = $('netPill');
    p.textContent = (on ? '● ' + U('online') + ' · ' : '✈ ') + U('offline'); p.className = 'pill' + (on ? '' : ' off');
  }
  window.addEventListener('beforeinstallprompt', (e) => { e.preventDefault(); st.installEvt = e; const b = $('installBtn'); b.hidden = false; b.onclick = async () => { b.hidden = true; st.installEvt.prompt(); }; });

  // ---------- voice ----------
  function stopSpeak() { if ('speechSynthesis' in window) speechSynthesis.cancel(); }
  function speak(text, onEnd) {
    const note = $('voiceNote');
    if (!('speechSynthesis' in window)) { if (note) note.textContent = U('no_voice'); return false; }
    speechSynthesis.cancel();
    const lang = VOICE[st.lang], u = new SpeechSynthesisUtterance(text); u.lang = lang; u.rate = 0.9;
    const v = speechSynthesis.getVoices().filter((x) => x.lang.replace('_', '-').toLowerCase().startsWith(lang.slice(0, 2)));
    if (note) note.textContent = v.length ? '' : U('no_voice');
    if (v.length) u.voice = v.find((x) => x.lang.toLowerCase().startsWith(lang.toLowerCase())) || v[0];
    if (onEnd) { u.onend = onEnd; u.onerror = onEnd; }
    speechSynthesis.speak(u); return true;
  }

  // ---------- image -> prediction ----------
  async function toBitmap(file) {
    if (window.createImageBitmap) { try { return await createImageBitmap(file, { imageOrientation: 'from-image' }); } catch (e) { /* fall through */ } }
    return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = URL.createObjectURL(file); });
  }
  function allowedIdx() {
    if (st.crop === 'auto') return null;
    const idx = []; st.model.m.classes.forEach((id, i) => { if (st.byId[id] && (st.byId[id].crop === st.crop || st.byId[id].crop === 'other')) idx.push(i); });
    return idx;
  }
  // average brightness and sharpness (variance of a 4-neighbour Laplacian on grey values)
  function photoQuality(px, n) {
    const g = new Float32Array(n * n); let lum = 0;
    for (let i = 0, j = 0; i < px.length; i += 4, j++) { g[j] = 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2]; lum += g[j]; }
    lum /= g.length;
    let s = 0, s2 = 0, c = 0;
    for (let y = 1; y < n - 1; y++) for (let x = 1; x < n - 1; x++) {
      const k = y * n + x, l = g[k - 1] + g[k + 1] + g[k - n] + g[k + n] - 4 * g[k]; s += l; s2 += l * l; c++;
    }
    return { lum, blur: s2 / c - (s / c) * (s / c) };
  }
  async function analyse(source, force) {
    show('busy'); await new Promise((r) => setTimeout(r, 60));
    try {
      const sw = source.width || source.naturalWidth, sh = source.height || source.naturalHeight, side = Math.min(sw, sh);
      const cv = $('work'), ctx = cv.getContext('2d', { willReadFrequently: true });
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(source, (sw - side) / 2, (sh - side) / 2, side, side, 0, 0, 256, 256);
      const px = ctx.getImageData(0, 0, 256, 256).data;
      const q = photoQuality(px, 256), lum = q.lum;
      const thumb = document.createElement('canvas'); thumb.width = thumb.height = 112;
      thumb.getContext('2d').drawImage(cv, 0, 0, 112, 112);
      const thumbUrl = thumb.toDataURL('image/jpeg', .7);
      if (!force) {
        const issue = lum < 45 ? 'coach_dark' : lum > 235 ? 'coach_bright' : q.blur < BLUR_MIN ? 'coach_blur' : null;
        if (issue) { st.pending = source; renderCoach(issue, thumbUrl); return; }
      }
      const x = PikEngine.preprocess(px, 256, 256, st.model.m);
      const ranked = PikEngine.classifyTTA(st.model, x, allowedIdx());
      const res = { t: Date.now(), crop: st.crop, lum, thumb: thumbUrl, top: ranked.slice(0, 3).map((o) => ({ id: st.model.m.classes[o.i], p: o.p })) };
      st.last = res; st.pending = null; saveHistory(res); renderResult(res, true);
    } catch (e) { console.error(e); show('home'); alert(U('error')); }
  }
  async function fromFile(f) { if (f) { try { await analyse(await toBitmap(f)); } catch (e) { alert(U('error')); } } }
  $('camInput').onchange = (e) => { fromFile(e.target.files[0]); e.target.value = ''; };
  $('galInput').onchange = (e) => { fromFile(e.target.files[0]); e.target.value = ''; };

  function renderSamples() {
    const box = $('samples'); box.innerHTML = '';
    SAMPLES.forEach((s) => {
      const b = document.createElement('button'); b.className = 'sample'; b.setAttribute('aria-label', 'sample ' + s);
      b.innerHTML = '<img src="samples/' + s + '.jpg" alt="" loading="lazy">';
      b.onclick = () => { const im = new Image(); im.onload = () => analyse(im, true); im.src = 'samples/' + s + '.jpg'; };
      box.appendChild(b);
    });
  }

  // ---------- photo coach (shown before diagnosing when the photo is dark, glaring or blurry) ----------
  function renderCoach(issue, thumbUrl) {
    const msg = U(issue);
    $('coach').innerHTML = '<div class="verdict v-grey"><span class="vicon">📷</span><div><div class="vbig">' + esc(U('retake')) + '</div></div></div>' +
      '<div class="card warn"><div class="hero"><img alt="" src="' + thumbUrl + '"><div><h2>' + esc(msg) + '</h2></div></div></div>' +
      '<div class="cta"><button class="btn fill" id="retakeBtn">📷 ' + esc(U('retake')) + '</button><button class="btn" id="useBtn">' + esc(U('use_anyway')) + '</button></div>';
    show('coach'); window.scrollTo(0, 0);
    $('retakeBtn').onclick = () => { stopSpeak(); $('camInput').click(); };
    $('useBtn').onclick = () => { stopSpeak(); if (st.pending) analyse(st.pending, true); };
    if (st.auto) speak(msg);
  }

  // ---------- result ----------
  function steps(ids) { return '<ul class="steps">' + ids.map((k) => '<li><span class="si">' + (ICON[k] || '•') + '</span><span>' + esc(tr(st.kb.snippets[k])) + '</span></li>').join('') + '</ul>'; }
  function dots(p) { const n = p >= 0.95 ? 5 : p >= 0.88 ? 4 : 3; let d = ''; for (let i = 0; i < 5; i++) d += '<i class="' + (i < n ? 'on' : '') + '"></i>'; return '<span class="dots" aria-hidden="true">' + d + '</span>'; }
  function renderResult(r, auto) {
    const m = st.model.m, top = r.top[0], c = st.byId[top.id], conf = Math.round(top.p * 100);
    const other = c.crop === 'other', sure = !other && top.p >= m.minConf;
    let h = '';
    // 1. traffic-light banner: the first thing a farmer sees and hears
    if (sure) {
      const u = c.urgency, col = u === 'high' ? 'red' : u === 'none' ? 'green' : 'amber', ic = { high: '🚨', medium: '⚠️', low: '👀', none: '✅' }[u];
      h += '<div class="verdict v-' + col + '"><span class="vicon">' + ic + '</span><div><div class="vbig">' + esc(tr(st.kb.ui.urgency[u])) + '</div></div></div>';
    } else {
      h += '<div class="verdict v-grey"><span class="vicon">' + (other ? '🚫' : '❓') + '</span><div><div class="vbig">' + esc(other ? U('other_title') : U('not_sure_title')) + '</div></div></div>';
    }
    // 2. what it is
    h += '<div class="card' + (sure ? '' : ' warn') + '"><div class="hero"><img alt="" src="' + r.thumb + '"><div>';
    if (sure) {
      h += '<h2>' + esc(tr(c.name)) + '</h2>';
      h += '<span class="badge b-type">' + (CROP_EMOJI[c.crop] || '') + ' ' + esc(tr(st.kb.crops[c.crop])) + ' · ' + esc(tr(st.kb.ui.types[c.type])) + '</span>';
      h += '<div class="small sure">' + dots(top.p) + ' ' + esc(top.p >= 0.9 ? U('sure_very') : U('sure_fair')) + '</div>';
    } else {
      h += '<h2>' + esc(other ? U('other_title') : U('not_sure_title')) + '</h2><p class="small">' + esc(other ? U('other_body') : U('not_sure_body')) + '</p>';
    }
    h += '</div></div></div>';
    // 3. what to do: big icon steps
    if (sure) h += '<div class="card sec"><h3>' + esc(U('what_do')) + '</h3>' + steps(c.do) + '</div>';
    // 4. actions a farmer can take right now
    h += '<div class="acts">';
    h += '<a class="abtn call" href="tel:' + KCC + '"><span>📞</span>' + esc(U('call_now')) + '</a>';
    h += '<button class="abtn" id="listenBtn"><span>🔊</span>' + esc(U('listen')) + '</button>';
    if (sure) h += '<button class="abtn" id="shopBtn"><span>🏪</span>' + esc(U('show_shop')) + '</button><button class="abtn" id="waBtn"><span>💬</span>' + esc(U('share_wa')) + '</button>';
    else h += '<button class="abtn" id="retryBtn"><span>📷</span>' + esc(U('retake')) + '</button>';
    h += '</div><p class="small" id="voiceNote"></p>';
    h += '<button class="btn fill wide" id="againBtn">📷 ' + esc(U('again')) + '</button>';
    // 5. everything else, tucked away
    h += '<details class="card more"><summary>' + esc(U('more_details')) + '</summary>';
    if (sure) {
      h += '<h3>' + esc(U('what_see')) + '</h3><p>' + esc(tr(c.see)) + '</p>';
      h += '<h3>' + esc(U('how_prevent')) + '</h3>' + steps(c.prevent);
      h += '<p class="small">' + esc(U('confidence')) + ': ' + conf + '%</p>';
    } else if (!other) {
      h += '<h3>' + esc(U('best_guess')) + '</h3><p>' + esc(tr(c.name)) + ' (' + esc(tr(st.kb.crops[c.crop])) + ')</p>';
    }
    const alts = r.top.slice(1).filter((o) => st.byId[o.id].crop !== 'other');
    if (!other && alts.length) h += '<h3>' + esc(U('other_possible')) + '</h3>' + alts.map((o) => { const k = st.byId[o.id]; return '<div class="alt"><span>' + esc(tr(k.name)) + ' <span class="small">(' + esc(tr(st.kb.crops[k.crop])) + ')</span></span><b>' + Math.round(o.p * 100) + '%</b></div>'; }).join('');
    h += '<p class="small">' + esc(U('kcc')) + '</p></details>';
    $('result').innerHTML = h; show('result'); window.scrollTo(0, 0);

    const text = sure ? speechFor(c) : other ? U('other_title') + '. ' + U('other_body') : U('not_sure_title') + '. ' + U('not_sure_body');
    const lb = $('listenBtn');
    lb.onclick = () => { if ('speechSynthesis' in window && speechSynthesis.speaking) { stopSpeak(); lb.lastChild.textContent = U('listen'); return; } lb.lastChild.textContent = U('stop'); speak(text, () => { lb.lastChild.textContent = U('listen'); }); };
    $('againBtn').onclick = () => { stopSpeak(); st.last = null; show('home'); };
    if ($('retryBtn')) $('retryBtn').onclick = () => { stopSpeak(); $('camInput').click(); };
    if (sure) { $('shopBtn').onclick = () => openShop(c, r); $('waBtn').onclick = () => shareWA(c); }
    if (auto && st.auto) { lb.lastChild.textContent = U('stop'); speak(text, () => { lb.lastChild.textContent = U('listen'); }); }
  }
  function speechFor(c) {
    const parts = [tr(c.name), tr(st.kb.crops[c.crop]), tr(st.kb.ui.urgency[c.urgency]), U('what_do')].concat(c.do.map((k) => tr(st.kb.snippets[k])));
    return parts.join('. ');
  }

  // ---------- hand-off: show to the shop / WhatsApp ----------
  function openShop(c, r) {
    const d = new Date(r.t).toLocaleDateString(st.lang === 'en' ? 'en-IN' : st.lang + '-IN');
    const two = (o) => esc(tr(o)) + (st.lang !== 'en' ? '<small>' + esc(en(o)) + '</small>' : '');
    $('shop').innerHTML = '<div class="shop-in"><p class="small">' + esc(U('shop_title')) + ' · ' + esc(d) + '</p>' +
      '<img alt="" src="' + r.thumb + '"><div class="shop-crop">' + two(st.kb.crops[c.crop]) + '</div>' +
      '<div class="shop-name">' + two(c.name) + '</div>' +
      '<div class="shop-ask">' + two(st.kb.ui.shop_ask) + '</div>' +
      '<p class="small">' + esc(U('shop_note')) + '</p>' +
      '<a class="abtn call" href="tel:' + KCC + '"><span>📞</span>' + esc(U('call_now')) + '</a>' +
      '<button class="btn fill wide" id="shopClose">' + esc(U('close')) + '</button></div>';
    $('shop').hidden = false; document.body.style.overflow = 'hidden';
    $('shopClose').onclick = () => { $('shop').hidden = true; document.body.style.overflow = ''; };
  }
  function shareWA(c) {
    if (!navigator.onLine) { alert(U('wa_offline')); return; }
    const t = U('wa_prefix') + ': ' + tr(st.kb.crops[c.crop]) + ' - ' + tr(c.name) + ' (' + tr(st.kb.ui.urgency[c.urgency]) + ')';
    window.open('https://wa.me/?text=' + encodeURIComponent(t), '_blank', 'noopener');
  }

  // ---------- history ----------
  function loadHistory() { try { return JSON.parse(lsGet('pik.history', '[]')); } catch (e) { return []; } }
  function saveHistory(r) { const h = loadHistory(); h.unshift(r); lsSet('pik.history', JSON.stringify(h.slice(0, 12))); renderHistory(); }
  function renderHistory() {
    const box = $('history'), h = loadHistory(); box.innerHTML = '';
    $('clearHist').hidden = !h.length;
    if (!h.length) { box.innerHTML = '<p class="small">' + esc(U('history_empty')) + '</p>'; return; }
    h.forEach((r) => {
      const c = st.byId[r.top[0].id]; if (!c) return;
      const b = document.createElement('button'); b.className = 'hitem';
      b.innerHTML = '<img alt="" src="' + esc(r.thumb) + '"><span><b>' + esc(tr(c.name)) + '</b><small>' + esc(tr(st.kb.crops[c.crop])) + ' · ' + new Date(r.t).toLocaleDateString(st.lang === 'en' ? 'en-IN' : st.lang + '-IN') + '</small></span>';
      b.onclick = () => { st.last = r; renderResult(r, false); }; box.appendChild(b);
    });
  }
  $('clearHist').onclick = () => { lsSet('pik.history', '[]'); renderHistory(); };

  function show(which) { ['home', 'busy', 'coach', 'result'].forEach((k) => { $(k).hidden = k !== which; }); }
  boot();
})();
