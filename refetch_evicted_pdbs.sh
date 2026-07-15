#!/usr/bin/env bash
# refetch_evicted_pdbs.sh — re-download iCloud-evicted PDB placeholders from RCSB.
# Writes to a .tmp then atomically moves over the dataless placeholder, so the
# result is a fully-local file (no read of the placeholder is ever needed).
set -uo pipefail
cd "$(dirname "$0")"

LIST="${1:-/tmp/dataless_pdbs.txt}"
FAILED=/tmp/pdb_refetch_failed.txt
: > "$FAILED"

fetch_one() {
  f="$1"
  [ -z "$f" ] && return 0
  id=$(basename "$f" .pdb | tr 'a-z' 'A-Z')
  for attempt in 1 2 3; do
    if curl -sf --max-time 30 -o "$f.tmp" "https://files.rcsb.org/download/$id.pdb" && [ -s "$f.tmp" ]; then
      mv "$f.tmp" "$f"
      return 0
    fi
    rm -f "$f.tmp"
    sleep 1
  done
  echo "$f" >> "$FAILED"
}
export -f fetch_one
export FAILED

total=$(wc -l < "$LIST" | tr -d ' ')
echo "Re-fetching $total evicted PDBs (parallel)..."
cat "$LIST" | xargs -P 12 -I {} bash -c 'fetch_one "$@"' _ {}

echo "DONE. failed: $(wc -l < "$FAILED" | tr -d ' ')"
