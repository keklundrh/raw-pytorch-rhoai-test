#!/usr/bin/env bash
# Download the gated weights for CONCH, TITAN, and Virchow2 into models/<name>/.
# Run locally with your approved Hugging Face login: python -m huggingface_hub.commands.huggingface_cli login
# Or set HF_TOKEN. The cluster runtime does not download weights or need a token.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_dir="${MODELS_DIR:-$root/models}"
mkdir -p "$models_dir"

python_bin="${PYTHON:-python}"
if ! "$python_bin" -c "import huggingface_hub" 2>/dev/null; then
    echo "huggingface_hub not importable with $python_bin; activate your venv or set PYTHON" >&2
    exit 1
fi

# repo_id:destination, all gated; your token needs access granted to each.
repos=(
    "MahmoodLab/CONCH:conch"
    "MahmoodLab/TITAN:titan"
    "paige-ai/Virchow2:virchow2"
)

for entry in "${repos[@]}"; do
    repo_id="${entry%%:*}"
    dest="${entry##*:}"
    dest_dir="$models_dir/$dest"
    if [ -d "$dest_dir" ] && [ -n "$(ls -A "$dest_dir" 2>/dev/null)" ]; then
        echo "$repo_id already present at $dest_dir; skipping (remove it to re-download)"
        continue
    fi
    echo "Downloading $repo_id -> $dest_dir"
    HF_HUB_DISABLE_PROGRESS_BARS=1 "$python_bin" - "$repo_id" "$dest_dir" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, dest_dir = sys.argv[1], sys.argv[2]
snapshot_download(
    repo_id=repo_id,
    local_dir=dest_dir,
    # Code files only; weights are downloaded with the same call.
    allow_patterns=[
        "*.json", "*.py", "*.txt", "*.md", "*.jpg", "*.png",
        "*.bin", "*.safetensors", "tokenizer*",
    ],
    max_workers=4,
)
print(f"Downloaded {repo_id} to {dest_dir}")
PY
    "$python_bin" - "$dest_dir" <<'PY'
import sys
from pathlib import Path

dest_dir = Path(sys.argv[1])
if dest_dir.name != "titan":
    sys.exit(0)

# Upstream TITAN remote code loads its tokenizer by Hub id, which fails on an
# offline serving pod. Patch conch_tokenizer.py to prefer the local snapshot
# via WEIGHTS_DIR. Re-applied after every download so a re-download cannot
# clobber the patch.
tokenizer_py = dest_dir / "conch_tokenizer.py"
text = tokenizer_py.read_text()
old = 'self.tokenizer = PreTrainedTokenizerFast.from_pretrained("MahmoodLab/TITAN")'
new = (
    'weights_dir = os.getenv("WEIGHTS_DIR")\n'
    '        if weights_dir and os.path.isfile(os.path.join(weights_dir, "titan", "tokenizer.json")):\n'
    '            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(\n'
    '                os.path.join(weights_dir, "titan"), local_files_only=True\n'
    '            )\n'
    '        else:\n'
    '            self.tokenizer = PreTrainedTokenizerFast.from_pretrained("MahmoodLab/TITAN")'
)
if old in text:
    tokenizer_py.write_text(text.replace(old, new, 1))
    print(f"Patched {tokenizer_py} for offline WEIGHTS_DIR loading")
elif 'os.getenv("WEIGHTS_DIR")' in text:
    print(f"{tokenizer_py} already patched")
else:
    print(f"WARNING: {tokenizer_py} no longer matches the expected upstream line; "
          f"verify it loads the tokenizer from the local snapshot" , file=sys.stderr)
PY
    "$python_bin" - "$repo_id" "$dest_dir" "$models_dir" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

repo_id, dest_dir, models_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
records = []
for weight in sorted(dest_dir.iterdir()):
    if weight.suffix not in (".bin", ".safetensors"):
        continue
    digest = hashlib.sha256()
    with weight.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    records.append({
        "repository": repo_id,
        "filename": f"{dest_dir.name}/{weight.name}",
        "sha256": digest.hexdigest(),
        "bytes": weight.stat().st_size,
    })
record = {"model": dest_dir.name, "files": records}
out = models_dir / "provenance.json"
try:
    existing = json.loads(out.read_text())
except (OSError, json.JSONDecodeError):
    existing = {}
existing[dest_dir.name] = record
out.write_text(json.dumps(existing, indent=2) + "\n")
print(f"Provenance for {dest_dir.name} recorded in {out}")
PY
done

echo "All model snapshots are under $models_dir; do not commit models/ to Git."
