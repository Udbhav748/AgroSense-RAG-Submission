// Client-side guardrails that mirror the backend's own limits (see
// backend/app/core/config.py). Kept in one place so a config change on
// either side only needs one edit here instead of hunting every
// hardcoded "20MB" across the app.
export const MAX_UPLOAD_SIZE_MB = 20
export const MAX_IMAGE_SIZE_MB = 20
export const MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
export const MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
