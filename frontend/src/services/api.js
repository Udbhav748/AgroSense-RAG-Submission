import axios from 'axios'

// The web frontend authenticates via individual user login (JWT) now,
// not the shared X-API-Key used by non-browser/service clients — see
// backend/app/core/auth.py's require_auth. Token lives in localStorage
// under the same key AuthContext reads/writes, read fresh on every
// request rather than captured once at module load, so logging in/out
// takes effect on the very next call with no need to recreate this
// instance.
export const AUTH_TOKEN_KEY = 'insightai_auth_token'

const api = axios.create({
  // Same-origin /api in dev: Vite proxies it to the backend (see
  // vite.config.js), so the browser never has to reach the backend host
  // directly or deal with CORS — works from localhost and LAN IPs alike.
  // Production builds set VITE_API_BASE_URL to the real backend origin.
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  timeout: 45000,
})

api.interceptors.request.use((config) => {
  const token = window.localStorage.getItem(AUTH_TOKEN_KEY)
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// A 401 means the token is missing/expired/invalid — clear it and send
// the user back to /login. A hard redirect (not react-router's
// navigate) because this file has no access to a router context, and a
// stale token is a full-app-state problem, not one page's concern.
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 && window.location.pathname !== '/login') {
      window.localStorage.removeItem(AUTH_TOKEN_KEY)
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export default api
