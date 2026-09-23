import api from './api'

/**
 * Fetch the backend's /health payload — liveness plus LLM-provider
 * readiness (booleans only, never a key) and which multi-modal RAG
 * capabilities the deployment has enabled (see
 * backend/app/api/v1/routes/health.py).
 * @returns {Promise<{status: string, llm: object, multimodal: object}>}
 */
export async function getHealthStatus() {
  const { data } = await api.get('/health')
  return data
}
