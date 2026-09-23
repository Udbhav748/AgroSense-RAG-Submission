/**
 * AgroSense-RAG Agronomic RAG & LeafSense - Service Worker
 * Progressive Web App (PWA) Offline & Field Resilience Engine
 *
 * Capabilities:
 * - Pre-caches core agronomic assets, app shells, and offline plant pathology lookup data.
 * - Provides offline runtime caching for static routes: /, /diagnose, /architecture, /documents, /chat.
 * - Stale-while-revalidate / Cache-first runtime asset caching for rural field resilience.
 */

const CACHE_VERSION = 'v1.0.0'
const STATIC_CACHE = `insightai-agronomy-static-${CACHE_VERSION}`
const RUNTIME_CACHE = `insightai-agronomy-runtime-${CACHE_VERSION}`
const DATA_CACHE = `insightai-pathology-data-${CACHE_VERSION}`

// Core static assets and app shells to pre-cache immediately on install
const PRECACHE_ASSETS = [
  '/',
  '/index.html',
  '/manifest.json',
  '/theme-init.js',
  '/data/offline-pathology.json',
  '/icons/icon-192x192.svg',
  '/icons/icon-512x512.svg',
  '/diagnose',
  '/architecture',
  '/documents',
  '/chat',
]

// 1. Install Event: Pre-cache core shell, routes, and agronomic pathology data
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(STATIC_CACHE)
      .then(async (cache) => {
        // Cache individual items gracefully so a failure in a single virtual route doesn't abort the entire install
        await Promise.allSettled(
          PRECACHE_ASSETS.map(async (url) => {
            try {
              const res = await fetch(url, { cache: 'reload' })
              if (res.ok) {
                await cache.put(url, res)
              }
            } catch {
              // Non-fatal during local dev or SSR preview
            }
          })
        )
      })
      .then(() => self.skipWaiting())
  )
})

// 2. Activate Event: Purge outdated cache versions & claim active clients
self.addEventListener('activate', (event) => {
  const currentCaches = [STATIC_CACHE, RUNTIME_CACHE, DATA_CACHE]

  event.waitUntil(
    caches
      .keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames.map((cacheName) => {
            if (!currentCaches.includes(cacheName)) {
              return caches.delete(cacheName)
            }
            return null
          })
        )
      })
      .then(() => self.clients.claim())
  )
})

// 3. Fetch Event: Multi-strategy caching for offline rural field operations
self.addEventListener('fetch', (event) => {
  const { request } = event
  const url = new URL(request.url)

  // Skip non-GET requests (e.g. file uploads, chat POSTs)
  if (request.method !== 'GET') {
    return
  }

  // A. Offline Plant Pathology & Agronomic Lookup Data
  if (url.pathname.includes('/data/offline-pathology.json') || url.pathname.includes('/api/pathology/offline')) {
    event.respondWith(
      caches.open(DATA_CACHE).then(async (cache) => {
        const cachedResponse = await cache.match(request)
        const fetchPromise = fetch(request)
          .then((networkResponse) => {
            if (networkResponse && networkResponse.status === 200) {
              cache.put(request, networkResponse.clone())
            }
            return networkResponse
          })
          .catch(() => cachedResponse)

        return cachedResponse || fetchPromise
      })
    )
    return
  }

  // B. Navigation requests (App Shell for SPA routes: /, /diagnose, /chat, etc.)
  if (request.mode === 'navigate') {
    event.respondWith(
      fetch(request)
        .then(async (networkResponse) => {
          if (networkResponse && networkResponse.status === 200) {
            const cache = await caches.open(RUNTIME_CACHE)
            cache.put(request, networkResponse.clone())
          }
          return networkResponse
        })
        .catch(async () => {
          // Offline fallback: check if specific route is cached, else fallback to /index.html
          const cachedRoute = await caches.match(request)
          if (cachedRoute) return cachedRoute

          const cachedShell = await caches.match('/index.html')
          if (cachedShell) return cachedShell

          const cachedRoot = await caches.match('/')
          return cachedRoot || Response.error()
        })
    )
    return
  }

  // C. Static assets (JS, CSS, Images, SVGs, Fonts)
  if (
    url.pathname.startsWith('/assets/') ||
    url.pathname.endsWith('.js') ||
    url.pathname.endsWith('.css') ||
    url.pathname.endsWith('.svg') ||
    url.pathname.endsWith('.png') ||
    url.pathname.endsWith('.woff2') ||
    url.pathname.endsWith('.woff') ||
    url.hostname.includes('fonts.googleapis.com') ||
    url.hostname.includes('fonts.gstatic.com')
  ) {
    event.respondWith(
      caches.match(request).then((cachedResponse) => {
        if (cachedResponse) {
          // Stale-while-revalidate: return cached response immediately, update cache in background
          fetch(request)
            .then(async (networkResponse) => {
              if (networkResponse && networkResponse.status === 200) {
                const cache = await caches.open(RUNTIME_CACHE)
                cache.put(request, networkResponse.clone())
              }
            })
            .catch(() => {})
          return cachedResponse
        }

        // Cache miss: fetch from network and cache
        return fetch(request)
          .then(async (networkResponse) => {
            if (networkResponse && networkResponse.status === 200) {
              const cache = await caches.open(RUNTIME_CACHE)
              cache.put(request, networkResponse.clone())
            }
            return networkResponse
          })
          .catch(() => {
            // Graceful fallback if offline
            return Response.error()
          })
      })
    )
    return
  }

  // D. Default network-first with cache fallback
  event.respondWith(
    fetch(request)
      .then(async (networkResponse) => {
        if (networkResponse && networkResponse.status === 200) {
          const cache = await caches.open(RUNTIME_CACHE)
          cache.put(request, networkResponse.clone())
        }
        return networkResponse
      })
      .catch(async () => {
        const cached = await caches.match(request)
        return cached || Response.error()
      })
  )
})

// Support message triggers (e.g. skipWaiting)
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting()
  }
})
