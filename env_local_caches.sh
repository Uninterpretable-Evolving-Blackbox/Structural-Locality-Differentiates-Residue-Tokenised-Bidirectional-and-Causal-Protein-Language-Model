# env_local_caches.sh — source this before any pipeline run.
# Redirects Python bytecode + numba JIT caches OUT of the iCloud-synced project
# tree. Writing/reading those caches inside iCloud stalls (dataless placeholders
# on read, fileproviderd contention on write), which hangs imports. Keep them
# on local /tmp so imports never touch iCloud-managed cache files.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NUMBA_CACHE_DIR=/tmp/own_sae_numba_cache
export PYTHONPYCACHEPREFIX=/tmp/own_sae_pycache
mkdir -p "$NUMBA_CACHE_DIR" "$PYTHONPYCACHEPREFIX"
