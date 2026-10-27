# Serve CONCH, TITAN, and Virchow2 on OpenShift AI 3.5

One custom ServingRuntime image serves all three models. Each model runs as its
own InferenceService pod from the same image, selected by `MODEL_NAME`, with the
shared weights on one PVC. This is a starting deployment, not a certified or
production-validated runtime. Changes are expected before production, to provide
GPU capabilities or facilitate deployments, like ModelCars or ONNX conversion. 

Another option: custom workbench image allowing users to load model and use similarly
to local notebooks and workflow. The down side is potentially multiple identical
models running within the environment.

## Why a custom ServingRuntime

Models are raw PyTorch. No OpenShift AI runtime can load these models: vLLM serves generative
LLMs, the OpenVINO Model Server and the MLServer runtime serve ONNX/OpenVINO IR
and pickle formats (not raw PyTorch; MLServer is Technology Preview), and
Triton's `pytorch` format means TorchScript. 

A custom ServingRuntime is the documented OpenShift AI extension point for 
arbitrary models (RawDeployment mode exists for exactly this) and is what this 
bundle uses. The alternative is porting the weights to ONNX (see below), which 
moves to preinstalled runtimes but shifts preprocessing to the client.


| Model | Pod endpoint | Input | Output |
|---|---|---|---|
| CONCH | `/v1/models/conch:predict` | base64 PNG/JPEG | 512-dim image embedding |
| TITAN | `/v1/models/titan:predict` | base64 HDF5 of pre-extracted CONCHv1.5 patch features | slide embedding |
| Virchow2 | `/v1/models/virchow2:predict` | base64 PNG/JPEG | 2560-dim tile embedding |

Envelope for all three (KServe v1-style):

```json
{"instances": [{"image": {"b64": "BASE64_BYTES"}}]}
```

For TITAN the `b64` bytes are an HDF5 file containing `features` shaped
`(1, N, C)`, `coords` shaped `(1, N, 2)`, and a `patch_size_level0` attribute on
`coords` (1024 if the slide is 40x, 512 if 20x). TITAN consumes pre-extracted
patch features, not images; extract them with CONCHv1.5 (CLAM or TRIDENT).

## A note on GPU support 

GPU support for models is possible but requires rebuilding custom serving runtime. 


## Files in this folder

| File | Purpose |
|---|---|
| `model.py` | Three model wrappers: CONCH, TITAN, Virchow2 |
| `server.py` | HTTP server; `MODEL_NAME` picks the model |
| `Containerfile` | Runtime image definition (CPU-only) |
| `serving-runtime.yaml` | One ServingRuntime template for all three models |
| `inference-service-conch.yaml` | CONCH deployment |
| `inference-service-titan.yaml` | TITAN deployment |
| `inference-service-virchow2.yaml` | Virchow2 deployment |
| `model-storage.yaml` | List of three per-model RWO PVCs |
| `model-upload-pod.yaml` | List of three temporary upload pods, one per PVC |
| `export_weights.sh` | Download the three gated model snapshots locally |
| `render_manifests.sh` | Render YAML with your project's registry image |
| `test_endpoint.sh` | Call a local or port-forwarded deployment |



## 1. Local setup and weight download

Prerequisites: your working venv with the CONCH test passing, `oc` logged in,
Hugging Face access granted to all three gated repos (they are CC-BY-NC-ND;
your HF account's primary email must match your institutional email for TITAN).

```bash
source .venv/bin/activate
python -m pip install -r openshift-ai/requirements-server.txt
./openshift-ai/export_weights.sh
```

The script downloads each repo snapshot into `models/<name>/` and records
revision hashes in `models/provenance.json`. Roughly 5 GiB total. Re-running
skips repos already present. Do not commit `models/` to Git.

## 2. Test the local HTTP server

Run one server per model, on separate ports:

```bash
MODEL_NAME=conch WEIGHTS_DIR="$PWD/models" python -m uvicorn server:app \
    --app-dir openshift-ai --host 127.0.0.1 --port 8080 --workers 1
```

```bash
MODEL_NAME=virchow2 WEIGHTS_DIR="$PWD/models" python -m uvicorn server:app \
    --app-dir openshift-ai --host 127.0.0.1 --port 8081 --workers 1
```

```bash
MODEL_NAME=titan WEIGHTS_DIR="$PWD/models" python -m uvicorn server:app \
    --app-dir openshift-ai --host 127.0.0.1 --port 8082 --workers 1
```

Wait for `Application startup complete`. In another terminal:

```bash
./openshift-ai/test_endpoint.sh conch local/sample.png
./openshift-ai/test_endpoint.sh virchow2 local/sample.png http://127.0.0.1:8081
./openshift-ai/test_endpoint.sh titan path/to/features.h5 http://127.0.0.1:8082
```

Expected: `SUCCESS: <model> returned an embedding`. For TITAN, download the
demo features from `MahmoodLab/TITAN` (`TCGA_demo_features/*.h5`) to exercise
the real path. Ctrl-C stops each server.

## 3. Select an OpenShift AI project

```bash
oc project YOUR_PROJECT
oc get crd servingruntimes.serving.kserve.io inferenceservices.serving.kserve.io
```

If a CRD is missing, ask your cluster administrator to enable the model serving
platform. The build must reach Red Hat's registry, PyPI, GitHub, and PyTorch's
wheel index. The running server does not download weights or need a token.

## 4. Build the runtime image on OpenShift

The cluster builds for its own node architecture. The image contains Python,
PyTorch CPU libraries, all three model wrappers, and the HTTP server — no
weights, tokens, or sample images.

```bash
oc new-build --name=pathology-runtime --binary=true --strategy=docker
oc start-build pathology-runtime --from-dir=openshift-ai --follow
```

Check the build actually succeeded — a `Failed` build leaves the previous image
on `:latest` and every later step runs stale code with no error:

```bash
oc get builds -o custom-columns=NAME:.metadata.name,PHASE:.status.phase,COMPLETED:.status.completionTimestamp
# The newest pathology-runtime-N must be Complete; if Failed: oc logs build/pathology-runtime-N
```

If a BuildConfig named `pathology-runtime` exists, skip `oc new-build`. Check
quota if the builder is evicted.

### Re-deploying after a code change

`oc apply` of the manifests does NOT roll pods or re-pull `:latest`: the
ServingRuntime spec is unchanged, so no new pod is created and the old pod keeps
the old image. The pod roll below is what deploys new code.
`imagePullPolicy: Always` (set in `serving-runtime.yaml`) only guarantees a pod
that IS being created re-pulls `:latest` from the registry instead of reusing
the node's cache — it never triggers a rollout by itself.

```bash
oc delete pod -l serving.kserve.io/inferenceservice=conch
oc delete pod -l serving.kserve.io/inferenceservice=titan
oc delete pod -l serving.kserve.io/inferenceservice=virchow2

# The pod's imageID must equal the istag digest (digest differs → pod is stale):
oc get pods -l serving.kserve.io/inferenceservice=titan \
    -o 'custom-columns=NAME:.metadata.name,IMAGE:.status.containerStatuses[0].imageID'
oc get istag pathology-runtime:latest -o jsonpath='{.image.dockerImageReference}{"\n"}'
```

To verify what code is inside the built image (independent of any pod):

```bash
oc run tmp-modelcheck --rm -i --restart=Never \
    --image=image-registry.openshift-image-registry.svc:5000/PROJECT/pathology-runtime:latest \
    -- grep -c _patch_dynamic_module_cache /opt/app-root/src/model.py
# prints 0 → the image predates the current model.py; rebuild
```



## 5. Put the weights on persistent cluster storage

One RWO PVC per model (`model-storage.yaml` is a List with three claims). Two
model pods can never Multi-Attach each other because no two of them share a
claim; only a leftover upload pod wedges its own model's PVC.

```bash
./openshift-ai/render_manifests.sh
oc apply -f rendered/model-storage.yaml
oc apply -f rendered/model-upload-pod.yaml
oc wait --for=condition=Ready pod/pathology-model-upload-conch --timeout=600s
oc wait --for=condition=Ready pod/pathology-model-upload-titan --timeout=600s
oc wait --for=condition=Ready pod/pathology-model-upload-virchow2 --timeout=600s
```

The three upload pods can run at once — they mount different volumes. Copy each
model directory to its own pod. `oc cp models/X pod:/models/X` nests inside the
existing path (creates `/models/X/X/`); the trailing `/.` avoids it:

```bash
oc cp models/conch/.    pathology-model-upload-conch:/models/conch/
oc cp models/titan/.    pathology-model-upload-titan:/models/titan/
oc cp models/virchow2/. pathology-model-upload-virchow2:/models/virchow2/
```

The loaders expect the snapshot under a `<model>/` subdirectory of the claim
(`$WEIGHTS_DIR/<model>/...`), so each PVC gets its model's directory even
though only one model lives on it.

Verify the FULL file set and checksums against `models/provenance.json` (one
file per PVC shown; run the same for `titan/` and `virchow2/`). A missing code
file (partial `oc cp`) produces the same startup traceback as a stale image, so
list the snapshot, not just the weight:

```bash
oc exec pathology-model-upload-conch -- ls /models/conch/
oc exec pathology-model-upload-titan -- ls /models/titan/
oc exec pathology-model-upload-virchow2 -- ls /models/virchow2/

oc exec pathology-model-upload-conch -- sha256sum /models/conch/pytorch_model.bin
oc exec pathology-model-upload-titan -- sha256sum /models/titan/conch_v1_5_pytorch_model.bin
oc exec pathology-model-upload-virchow2 -- sha256sum /models/virchow2/model.safetensors
```

Required on the titan PVC: `config.json`, `conch_tokenizer.py`, `conch_v1_5.py`,
`modeling_titan.py`, `configuration_titan.py`, `text_transformer.py`,
`vision_transformer.py`, `conch_v1_5_pytorch_model.bin` (`text_transformer.py`
imports `conch_tokenizer.py`; if either is missing the dynamic-module loader
fails at startup).

If the hashes match, remove the upload pods BEFORE deploying any model. Each
PVC is ReadWriteOnce: a running upload pod holds its volume's single
attachment and the matching model pod wedges in Multi-Attach until it is
deleted.

```bash
oc delete pod pathology-model-upload-conch pathology-model-upload-titan \
    pathology-model-upload-virchow2 --wait=true
```

If a model pod starts CrashLoopBackOff with `Model checkpoint not found` while
the upload pods are already deleted: first scale the model's InferenceService
to 0 (or delete it) so it releases the PVC — a re-created upload pod wedges in
Multi-Attach until then. Then re-create the upload pods (`oc apply -f
rendered/model-upload-pod.yaml`), wait for Ready, `mv` the flat files into the
`<model>/` subdirectory, delete the upload pod, and re-apply the
InferenceService.

The claims rely on the default storage class. If there is none, add your
administrator's `storageClassName` to each PVC.

### Inspecting a model PVC while its pod owns the volume

The model PVC is ReadWriteOnce: while the model's pod exists (even
crash-looping), a second pod mounting the claim wedges in Multi-Attach. To
check PVC contents, scale the model's InferenceService to 0 first, then run a
temporary pod that mounts the claim (paste-safe heredoc; `--overrides` JSON is
not):

```bash
oc scale inferenceservice titan --replicas=0

cat <<'EOF' | oc apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: tmp-pvc-ls
spec:
  restartPolicy: Never
  containers:
    - name: c
      image: image-registry.openshift-image-registry.svc:5000/PROJECT/pathology-runtime:latest
      command: ["ls", "-la", "/mnt/models/titan/"]
      volumeMounts:
        - name: models
          mountPath: /mnt/models
  volumes:
    - name: models
      persistentVolumeClaim:
        claimName: pathology-models-titan
EOF

oc logs tmp-pvc-ls          # the listing; retry once if empty (pod may still be starting)
oc delete pod tmp-pvc-ls
oc scale inferenceservice titan --replicas=1
```

Do NOT `oc wait --for=condition=Ready` on the tmp pod: a one-shot pod that
exits goes to `Succeeded` and never reports Ready, so the wait burns its full
timeout.


## 6. Deploy the served models

```bash
oc apply --dry-run=server -f rendered/serving-runtime.yaml
oc apply -f rendered/serving-runtime.yaml
oc apply --dry-run=server -f rendered/inference-service-conch.yaml
oc apply -f rendered/inference-service-conch.yaml
oc apply --dry-run=server -f rendered/inference-service-titan.yaml
oc apply -f rendered/inference-service-titan.yaml
oc apply --dry-run=server -f rendered/inference-service-virchow2.yaml
oc apply -f rendered/inference-service-virchow2.yaml
```

Readiness includes a real warm-up. Watch status with:

```bash
oc get inferenceservices conch titan virchow2
oc logs -l serving.kserve.io/inferenceservice=conch -c kserve-container --tail=100
```

If the pod is crash-looping with a startup traceback, first confirm it runs the
just-built image — see "Re-deploying after a code change" in step 4. A stale
`:latest` pod reproduces old tracebacks exactly.

Common failures:

- `ImagePullBackOff`: verify build completion and the rendered image path.
- `FileNotFoundError`: verify each PVC contains its model's snapshot under a
  `<model>/` subdirectory at the claim root (`/mnt/models/conch/...` on the
  conch PVC, `/mnt/models/titan/...` on the titan PVC, and so on).
- `MultiAttach` or pod stuck Pending with `RWO`: the model's own PVC is
  attached to another pod (e.g. its upload pod still running) — delete that
  pod. Model pods cannot Multi-Attach each other with per-model PVCs.
- dependency/import failure: compare package versions against your working Mac
  environment. Linux dependencies are independently resolved.
- `OOMKilled`: inspect actual memory use before raising the memory limit.
- admission error: use the exact server dry-run message to reconcile cluster
  policies.

## 7. Call the cluster deployments

Each service is cluster-local; no public Route is created. Use `oc port-forward`
per model on distinct local ports:

```bash
oc get pods -l serving.kserve.io/inferenceservice=conch
oc port-forward pod/PREDICTOR_POD_NAME 8081:8080
```

```bash
./openshift-ai/test_endpoint.sh conch local/sample.png http://127.0.0.1:8081
```

## Resource notes

- One model per pod keeps only one model copy resident per deployment; the
  runtime never loads a second model in a serving process.
- CPU baseline requests 2 CPUs / 4 GiB with an 8 GiB limit per pod. Virchow2
  (ViT-Huge) is slow on CPU — seconds per tile. Measure before raising limits.
- TITAN warm-up runs a tiny synthetic feature grid; real slide embeddings scale
  with patch count N.
- Each model has its own RWO PVC (~2 GiB data + overhead); model pods on
  different nodes never share a volume, so all three can run concurrently.
- These are starting allocations, not measured production requirements.

## Before calling this production-ready

1. Pin and scan the validated Linux dependencies, the CONCH Git commit, TITAN's
   transformers version, and image digests. Current requirements bootstrap the
   first deployment; they are not reproducible production locks.
2. Configure TLS and token authentication using the cluster's supported OpenShift
   AI ingress/auth setup before enabling external access.
3. Measure latency, memory, and throughput with real payloads. This CPU-only
   image does not become GPU-enabled by adding a GPU resource request.
   GPU support IS possible as a future option: OpenShift AI 3.5's accelerator
   support (NFD + NVIDIA/AMD GPU operator, hardware profiles) is
   runtime-agnostic, so this ServingRuntime can request `nvidia.com/gpu` — but
   the image must first be rebuilt with CUDA PyTorch wheels (the Containerfile
   currently installs CPU-only wheels), and the wrappers' hardcoded
   `.to("cpu")` calls need a device env var. Until then, the pod silently runs
   on CPU even when a GPU is attached.
4. Storage per model is already isolated (one RWO PVC each). Consider S3
   delivery or an RWX class before running multiple replicas of one model.
5. Add request/error/latency metrics and alerts; the server has health probes
   but no Prometheus application metrics endpoint.
6. Confirm model-use permissions. All three models are gated CC-BY-NC-ND for
   non-commercial academic research; academic access does not itself authorize
   customer, commercial, or clinical use.

## Porting to ONNX (optional, for out-of-the-box OpenShift AI runtimes)

The weights can be ported to ONNX to use the preinstalled OpenVINO Model Server
runtime instead of this custom image: no custom ServingRuntime, no image build,
standard KServe v2 tensor endpoint. The tradeoff: preprocessing (resize +
normalize) moves to the client, since OVMS serves a pure tensor model.

Port difficulty differs by model:

| Model | ONNX port | Why |
|---|---|---|
| CONCH | Easy | Fixed-shape ViT-B/16 + plain MultiheadAttention pooler, no data-dependent control flow |
| Virchow2 | Easy | timm ViT-Huge, class + mean pool |
| TITAN | Hard, unvalidated | Variable patch count N through attention pooling; needs its own parity validation |

Procedure (per model, CONCH shown):

1. Export the encoder only — the server's `encode_image` path is a pure tensor
   pipeline:

   ```python
   from conch.open_clip_custom import create_model_from_pretrained
   import torch

   model, _ = create_model_from_pretrained("conch_ViT-B-16", "models/conch/pytorch_model.bin")
   model.eval()

   class ImageEncoder(torch.nn.Module):
       def __init__(self, m):
           super().__init__()
           self.visual = m.visual
       def forward(self, x):  # (1, 3, 448, 448) float32
           return self.visual.forward_no_head(x, normalize=False)  # (1, 512)

   torch.onnx.export(
       ImageEncoder(model).eval(), torch.randn(1, 3, 448, 448), "conch_visual.onnx",
       input_names=["image"], output_names=["embedding"],
       dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
       opset_version=17, dynamo=False,
   )
   ```

2. **Validate parity before trusting the port**: run the same image through
   PyTorch and ONNX Runtime; the 512-dim embeddings must match (max abs diff
   < ~1e-4). OAI does not support a miscalibrated conversion, and the ONNX
   artifact inherits the model's CC-BY-NC-ND terms.

3. Store the ONNX file (S3 connection or OCI modelcar), deploy via the
   dashboard wizard with model format `onnx`, and call the documented
   `/v2/models/<name>/infer` endpoint with JSON tensors. The client must apply
   CONCH's preprocessing itself (448×448 resize, ImageNet mean/std).

Keep this custom runtime as the reference implementation until parity is
proven; cut over per model. TITAN can stay on this runtime regardless.

## Validation performed on this bundle
CONCH and TITAN were exercised against real weights: local HTTP server,
inference, the 512-dim CONCH embedding, and the 768-dim TITAN slide embedding
from the `TCGA_demo_features` h5. Python syntax, YAML parsing, and HTTP contract
tests with fake models cover PNG decoding, the h5 path, malformed payload
rejection, readiness, and failure to start without weights. This first
iteration has no request-size or image-size validation.

TITAN-specific notes:

- TITAN's remote code needs `einops`, `einops-exts`, and `timm` beyond the
  server requirements (`requirements-server.txt`).
- `export_weights.sh` patches `conch_tokenizer.py` in the TITAN snapshot so the
  tokenizer loads from the local snapshot via `WEIGHTS_DIR` instead of the Hub
  id; an offline serving pod cannot fetch `MahmoodLab/TITAN`.
- The synthetic warm-up h5 uses a 2x2 grid of non-zero 768-dim patch features;
  all-zero features trip TITAN's background-mask logic at startup.

NOT yet exercised: the Virchow2 load path against real weights, the Linux
container build, cluster admission, and deployed endpoints. After running
`export_weights.sh`, do a one-off local server start for each model (step 2)
before deploying, so a bad load path surfaces before OpenShift readiness
timeouts.

## Sources

- CONCH: https://huggingface.co/MahmoodLab/CONCH
- TITAN (slide embedding from patch features): https://huggingface.co/MahmoodLab/TITAN and https://github.com/mahmoodlab/TITAN
- Virchow2 (timm loading, class + mean patch token): https://huggingface.co/paige-ai/Virchow2
- OpenShift AI 3.5 runtimes and deployment modes: https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/deploying_models/deploying_models
- KServe PVC model storage pattern: https://kserve.github.io/website/docs/model-serving/storage/providers/pvc
