import sys
import time

import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from conch.open_clip_custom import create_model_from_pretrained

print("Downloading weights, or using the cached copy...")
checkpoint = hf_hub_download(
    repo_id="MahmoodLab/CONCH",
    filename="pytorch_model.bin",
)

print("Loading CONCH on the CPU...")
model, preprocess = create_model_from_pretrained(
    "conch_ViT-B-16",
    checkpoint,
)
model = model.to("cpu").eval()

if len(sys.argv) > 1:
    image = Image.open(sys.argv[1]).convert("RGB")
    print("Using image:", sys.argv[1])
else:
    image = Image.new("RGB", (448, 448), color=(180, 120, 160))
    print("Using a dummy image to check that the model runs.")

# Prepare one image in the format CONCH expects.
image_tensor = preprocess(image).unsqueeze(0).to("cpu")

print("Running inference...")
start = time.perf_counter()

with torch.inference_mode():
    embedding = model.encode_image(
        image_tensor,
        proj_contrast=False,
        normalize=False,
    )

if not torch.isfinite(embedding).all():
    raise RuntimeError("The model returned invalid numerical values.")

print("SUCCESS: CONCH returned an embedding.")
print("Output shape:", tuple(embedding.shape))
print("First five numbers:", embedding[0, :5].tolist())
print(f"Inference time: {time.perf_counter() - start:.2f} seconds")
