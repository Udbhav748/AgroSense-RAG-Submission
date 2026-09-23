#!/bin/sh
set -e

# AWS_LAMBDA_FUNCTION_NAME is a Lambda-reserved env var, always set inside a
# real Lambda execution environment and never set locally or in
# docker-compose.yml — so this whole block is a no-op outside Lambda.
if [ -n "$AWS_LAMBDA_FUNCTION_NAME" ]; then
  # Lambda's container filesystem is read-only everywhere except /tmp —
  # including the empty uploads/vector_store/feedback directories already
  # baked into the image, which can't even be deleted at runtime to make
  # room for a symlink (an earlier version of this script tried exactly
  # that: `rm -rf /app/uploads && ln -s ...`, and the rm failed with
  # "Read-only file system", which combined with `set -e` above crashed
  # the container before uvicorn ever started). Settings.data_dir()
  # (app/core/config.py) is the actual fix — DATA_DIR_OVERRIDE=/tmp below
  # makes upload_service.py/faiss_vector_store.py/feedback_service.py
  # resolve their directories under /tmp directly, so nothing here needs
  # to touch /app at all.
  export DATA_DIR_OVERRIDE=/tmp

  # huggingface_hub writes a filelock next to cached model weights even on
  # a pure read — fails under /app's now-read-only filesystem even though
  # the model (baked into the image at build time) needs no re-download.
  # Copying the already-baked cache into /tmp once per cold start keeps
  # the model available locally while giving the lock file somewhere
  # writable to live.
  mkdir -p /tmp/hf_cache
  cp -r /app/.cache/huggingface /tmp/hf_cache/huggingface
  export HF_HOME=/tmp/hf_cache/huggingface
  export TRANSFORMERS_CACHE=/tmp/hf_cache/huggingface
  export SENTENCE_TRANSFORMERS_HOME=/tmp/hf_cache/huggingface

  # Deliberately NOT setting HF_HUB_OFFLINE=1 here (an earlier version of
  # this script did). Verified directly: with it set, transformers'
  # AutoConfig.from_pretrained -> cached_files() offline-resolution path
  # raised LocalEntryNotFoundError ("we couldn't connect... and couldn't
  # find them in the cached files") even though the copied cache was
  # confirmed byte-for-byte identical to the source, including intact
  # blob symlinks and the correct refs/main revision pointer — an
  # offline-cache-resolution quirk in this transformers/huggingface_hub
  # version, not a real "no network" situation. Lambda functions outside
  # a VPC (this one is) have normal internet access, so the default,
  # already-proven-working path — a quick network check, then load from
  # the local cache, matching local dev's own behavior exactly — is both
  # simpler and correct. Confirmed working via direct test.
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
