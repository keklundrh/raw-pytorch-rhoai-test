#!/usr/bin/env bash
# Render the YAML templates for the current project's internal image registry.
# Requires a selected OpenShift project (oc project). Outputs to rendered/.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v oc >/dev/null 2>&1; then
    echo "oc not found; log in to the cluster first" >&2
    exit 1
fi

namespace="$(oc project -q 2>/dev/null || true)"
if [ -z "$namespace" ]; then
    echo "Select a valid OpenShift project first: oc project YOUR_PROJECT" >&2
    exit 1
fi
if ! printf '%s' "$namespace" | grep -Eq '^[a-z0-9]([-a-z0-9]*[a-z0-9])?$'; then
    echo "Not a valid namespace: $namespace" >&2
    exit 1
fi

image="image-registry.openshift-image-registry.svc:5000/${namespace}/pathology-runtime:latest"

mkdir -p "$root/rendered"
for name in model-storage.yaml model-upload-pod.yaml serving-runtime.yaml \
            inference-service-conch.yaml inference-service-titan.yaml \
            inference-service-virchow2.yaml; do
    sed "s|REPLACE_WITH_RUNTIME_IMAGE|$image|g" \
        "$root/openshift-ai/$name" > "$root/rendered/$name"
done

echo "Rendered manifests for namespace $namespace using $image"
