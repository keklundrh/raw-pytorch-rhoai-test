#!/usr/bin/env bash
# Send one payload to a local or port-forwarded model server.
#   test_endpoint.sh conch local/sample.png
#   test_endpoint.sh virchow2 local/sample.png [URL]
#   test_endpoint.sh titan features.h5 [URL]
# The model name must match the deployment being called.
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 {conch|titan|virchow2} {image.png|features.h5} [URL]" >&2
    echo "       URL defaults to http://127.0.0.1:8080" >&2
    exit 1
fi

model="$1"
payload_file="$2"
url="${3:-http://127.0.0.1:8080}"

case "$model" in
    conch|virchow2|titan) ;;
    *)
        echo "model must be conch, titan, or virchow2" >&2
        exit 1
        ;;
esac

if [ ! -f "$payload_file" ]; then
    echo "payload file not found: $payload_file" >&2
    exit 1
fi

if ! command -v base64 >/dev/null 2>&1; then
    echo "base64 not found" >&2
    exit 1
fi

b64="$(base64 < "$payload_file" | tr -d '\n')"
body="{\"instances\":[{\"image\":{\"b64\":\"$b64\"}}]}"
tmp_body="$(mktemp)"
trap 'rm -f "$tmp_body"' EXIT
printf '%s' "$body" > "$tmp_body"

# Guard against enormous command lines via argv: pass the body as a file.
curl_args=(--fail-with-body --silent --show-error --max-time 600
           -H "Content-Type: application/json"
           --data-binary "@$tmp_body")
if [ -n "${INFERENCE_TOKEN:-}" ]; then
    curl_args+=(-H "Authorization: Bearer $INFERENCE_TOKEN")
fi

echo "POST $url/v1/models/$model:predict ($(( $(wc -c < "$tmp_body") / 1024 )) KiB body)"
response="$(curl "${curl_args[@]}" "$url/v1/models/$model:predict")"

# Print embedding summary without dumping every value.
MODEL="$model" RESPONSE="$response" python3 - <<'PY'
import json, os, sys

response = os.environ["RESPONSE"]
try:
    result = json.loads(response)
    prediction = result["predictions"][0]
except (json.JSONDecodeError, KeyError, IndexError, TypeError):
    sys.exit(f"Unexpected response: {response[:400]}")
print(f"SUCCESS: {os.environ['MODEL']} returned an embedding")
print("Dimensions:", len(prediction))
print("First five values:", prediction[:5])
PY
