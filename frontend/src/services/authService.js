import api from './api'

/**
 * Create an account. consent must be true — the backend rejects a
 * missing/false value with 422 (see backend SignupRequest.consent).
 * @param {string} email
 * @param {string} password
 * @param {boolean} consent
 * @returns {Promise<{access_token: string, token_type: string}>}
 */
export async function signup(email, password, consent) {
  const { data } = await api.post('/auth/signup', { email, password, consent })
  return data
}

/**
 * @param {string} email
 * @param {string} password
 * @returns {Promise<{access_token: string, token_type: string}>}
 */
export async function login(email, password) {
  const { data } = await api.post('/auth/login', { email, password })
  return data
}

/**
 * Current caller's identity — used to hydrate auth state from a stored
 * token on page load, and to confirm a token is still valid.
 * @returns {Promise<{email: string, tenant_id: number, role: string}>}
 */
export async function getMe() {
  const { data } = await api.get('/auth/me')
  return data
}

/**
 * Revoke every token issued to the caller server-side (see backend's
 * POST /auth/logout) — real logout, not just discarding the token
 * locally. AuthContext calls this best-effort: the local token is
 * cleared either way, so a network failure here never blocks logging
 * out on this device, it just means other devices' sessions (if any)
 * stay live until their token naturally expires.
 * @returns {Promise<void>}
 */
export async function logout() {
  await api.post('/auth/logout')
}
