"""End-to-end: load online (installs service worker + caches model), then go OFFLINE, reload and run diagnoses."""
import threading, http.server, socketserver, functools, sys, os, json
from playwright.sync_api import sync_playwright
ROOT = os.path.join(os.path.dirname(__file__), '..', 'app'); SHOTS = os.path.join(os.path.dirname(__file__), '..', 'docs', 'screens'); os.makedirs(SHOTS, exist_ok=True)
class Q(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=ROOT, **k)
    def log_message(self, *a): pass
socketserver.TCPServer.allow_reuse_address = True
srv = socketserver.TCPServer(('127.0.0.1', 8765), Q); threading.Thread(target=srv.serve_forever, daemon=True).start()
errs = []; results = {}
with sync_playwright() as p:
    b = p.chromium.launch(); ctx = b.new_context(viewport={'width': 390, 'height': 800}, device_scale_factor=2, locale='en-IN')
    pg = ctx.new_page(); pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None); pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto('http://127.0.0.1:8765/'); pg.wait_for_function("document.getElementById('modelPill').textContent.includes('✓')", timeout=20000)
    pg.evaluate("navigator.serviceWorker.ready"); pg.wait_for_timeout(1500)
    ctx.set_offline(True); pg.reload(); pg.wait_for_function("document.getElementById('modelPill').textContent.includes('✓')", timeout=20000)
    print('offline reload OK; net pill =', pg.inner_text('#netPill'))
    pg.screenshot(path=f'{SHOTS}/home_en.png')
    for n in range(1, 7):
        import time; t0 = time.time(); pg.click(f'.sample >> nth={n-1}'); pg.wait_for_selector('#result:not([hidden])', timeout=15000); print('   inference+render ms:', int((time.time()-t0)*1000))
        title = pg.inner_text('#result h2'); conf = pg.inner_text('#result .small'); results[n] = (title, conf)
        print(f'sample s{n}:', title, '|', conf.split('\n')[0])
        if n == 1: pg.screenshot(path=f'{SHOTS}/result_en.png', full_page=True)
        pg.click('#againBtn')
    # out-of-scope inputs should come back as 'not sure' (or at least low confidence)
    import glob
    for f in [sorted(glob.glob('/home/claude/work/ood/Apple___healthy/*'))[0], sorted(glob.glob('/home/claude/work/ood/Strawberry___healthy/*'))[0], os.path.join(SHOTS, 'home_en.png')]:
        pg.set_input_files('#galInput', f); pg.wait_for_selector('#result:not([hidden])', timeout=15000)
        print('non-target input', os.path.basename(f)[:30], '->', pg.inner_text('#result h2'), '|', pg.inner_text('#result .small').split(chr(10))[-1]); pg.click('#againBtn')
    # languages
    pg.click('.sample >> nth=1'); pg.wait_for_selector('#result:not([hidden])')
    for lg in ['mr', 'hi']:
        pg.click(f'[data-lang={lg}]'); pg.wait_for_timeout(200); pg.screenshot(path=f'{SHOTS}/result_{lg}.png', full_page=True); print(lg, '->', pg.inner_text('#result h2'))
    pg.click('#againBtn'); pg.click('[data-lang=mr]'); pg.screenshot(path=f'{SHOTS}/home_mr.png')
    # crop filter + history persisted
    print('history items:', pg.locator('.hitem').count())
    b.close()
srv.shutdown()
print('console errors:', [e for e in errs if 'favicon' not in e] or 'none')
