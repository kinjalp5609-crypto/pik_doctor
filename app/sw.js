/* Pik Doctor service worker: cache-first so the whole app, model and advice work with no network. */
const VERSION = 'pik-v3';
const CORE = ['./', 'index.html', 'style.css', 'app.js', 'engine.js', 'manifest.webmanifest', 'data/kb.json', 'model/manifest.json', 'model/weights.bin',
  'icons/icon-192.png', 'icons/icon-512.png', 'samples/s1.jpg', 'samples/s2.jpg', 'samples/s3.jpg', 'samples/s4.jpg', 'samples/s5.jpg', 'samples/s6.jpg'];
self.addEventListener('install', (e) => { e.waitUntil(caches.open(VERSION).then((c) => c.addAll(CORE)).then(() => self.skipWaiting())); });
self.addEventListener('activate', (e) => { e.waitUntil(caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== VERSION).map((k) => caches.delete(k)))).then(() => self.clients.claim())); });
self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;
  e.respondWith(caches.match(e.request, { ignoreSearch: true }).then((hit) => hit || fetch(e.request).then((r) => {
    if (r.ok && new URL(e.request.url).origin === location.origin) { const copy = r.clone(); caches.open(VERSION).then((c) => c.put(e.request, copy)); }
    return r;
  }).catch(() => caches.match('index.html'))));
});
