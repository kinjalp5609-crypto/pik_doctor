"""UI end-to-end (farmer-friendly build): offline reload, traffic-light result, shop card, photo coach (blurred / dark photos)."""
import threading, http.server, socketserver, os, cv2
ROOT = os.path.join(os.path.dirname(__file__), '..', 'app'); SHOTS = os.path.join(os.path.dirname(__file__), '..', 'docs', 'screens')
im = cv2.imread(os.path.join(ROOT, 'samples', 's2.jpg')); cv2.imwrite('/tmp/blur.jpg', cv2.GaussianBlur(im, (0, 0), 3)); cv2.imwrite('/tmp/dark.jpg', (im * 0.12).astype('uint8'))
class Q(http.server.SimpleHTTPRequestHandler):
    def __init__(s, *a, **k): super().__init__(*a, directory=ROOT, **k)
    def log_message(s, *a): pass
socketserver.TCPServer.allow_reuse_address = True
srv = socketserver.TCPServer(('127.0.0.1', 8766), Q); threading.Thread(target=srv.serve_forever, daemon=True).start()
from playwright.sync_api import sync_playwright
errs = []
with sync_playwright() as p:
    b = p.chromium.launch(); ctx = b.new_context(viewport={'width': 390, 'height': 800}, device_scale_factor=2, locale='mr-IN')
    pg = ctx.new_page(); pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None); pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto('http://127.0.0.1:8766/'); pg.wait_for_function("document.getElementById('modelPill').textContent.includes('✓')", timeout=20000)
    pg.evaluate("navigator.serviceWorker.ready"); pg.wait_for_timeout(1500)
    ctx.set_offline(True); pg.reload(); pg.wait_for_function("document.getElementById('modelPill').textContent.includes('✓')", timeout=20000)
    print('offline reload ok |', pg.inner_text('#netPill')); pg.screenshot(path=f'{SHOTS}/home_mr.png')
    for n in range(6):
        pg.click(f'.sample >> nth={n}'); pg.wait_for_selector('#result:not([hidden])', timeout=15000)
        print(f's{n+1}:', pg.inner_text('#result .vbig'), '|', pg.inner_text('#result h2'))
        if n == 1: pg.screenshot(path=f'{SHOTS}/result_mr.png', full_page=True)
        pg.click('#againBtn')
    pg.click('.sample >> nth=1'); pg.wait_for_selector('#result:not([hidden])')
    pg.click('#shopBtn'); pg.wait_for_timeout(200); pg.screenshot(path=f'{SHOTS}/shop_mr.png'); pg.click('#shopClose')
    assert pg.get_attribute('#result .abtn.call', 'href') == 'tel:18001801551'
    pg.click('#againBtn')
    for f in ['/tmp/blur.jpg', '/tmp/dark.jpg']:
        pg.set_input_files('#galInput', f); pg.wait_for_selector('#coach:not([hidden])', timeout=15000)
        print(os.path.basename(f), '-> coach:', pg.inner_text('#coach h2'))
        pg.click('#useBtn'); pg.wait_for_selector('#result:not([hidden])'); print('   use anyway ->', pg.inner_text('#result h2')); pg.click('#againBtn')
    b.close()
srv.shutdown(); print('console errors:', errs or 'none')
