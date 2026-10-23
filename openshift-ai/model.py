"""Inference wrappers for CONCH, TITAN, and Virchow2. No downloads here.

Each wrapper owns everything model-specific: how to load the weights, how to
decode the request payload, and what sample to run for the warm-up.
"""
import io

from PIL import Image
from pathlib import Path
import os

MODELS = ("conch", "titan", "virchow2")


class ConchModel:
    """CONCH ViT-B/16 image embedding from a PNG/JPEG."""

    name = "conch"
    payload_kind = "image"

    def __init__(self, weights_dir):
        import torch
        from conch.open_clip_custom import create_model_from_pretrained

        checkpoint = Path(weights_dir) / "conch" / "pytorch_model.bin"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
        self.torch = torch
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))
        self.model, self.preprocess = create_model_from_pretrained(
            "conch_ViT-B-16", str(checkpoint)
        )
        self.model = self.model.to("cpu").eval()

    @staticmethod
    def warmup_payload():
        return Image.new("RGB", (448, 448))

    def predict(self, image):
        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to("cpu")
        with self.torch.inference_mode():
            embedding = self.model.encode_image(
                tensor, proj_contrast=False, normalize=False
            )
        if not self.torch.isfinite(embedding).all():
            raise RuntimeError("CONCH returned non-finite values")
        return embedding[0].tolist()


class TitanModel:
    """TITAN slide embedding from pre-extracted CONCHv1.5 patch features (HDF5)."""

    name = "titan"
    payload_kind = "h5"
    @staticmethod
    def _patch_dynamic_module_cache(repo):
        """transformers 4.x copies only the entry file's DIRECT relative imports
        into its dynamic-module cache for a local directory, but hashes the
        RECURSIVE closure. TITAN's text_transformer.py imports .conch_tokenizer,
        which is therefore missing at hash time. Copy any snapshot .py not yet
        in the cache so the closure is complete."""
        import shutil
        from transformers.dynamic_module_utils import HF_MODULES_CACHE

        cache_dir = Path(HF_MODULES_CACHE) / "transformers_modules" / "titan"
        cache_dir.mkdir(parents=True, exist_ok=True)
        for py in sorted(Path(repo).glob("*.py")):
            target = cache_dir / py.name
            if not target.is_file() or target.read_bytes() != py.read_bytes():
                shutil.copy(py, target)

    def __init__(self, weights_dir):
        import torch
        from transformers import AutoModel

        repo = Path(weights_dir) / "titan"
        if not (repo / "config.json").is_file():
            raise FileNotFoundError(f"TITAN snapshot not found: {repo}")
        self.torch = torch
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))
        self._patch_dynamic_module_cache(repo)
        self.model = AutoModel.from_pretrained(
            str(repo), trust_remote_code=True, local_files_only=True
        )
        self.model = self.model.to("cpu").eval()

    @staticmethod
    def warmup_payload():
        return _warmup_h5()

    def predict(self, h5_bytes):
        import h5py

        with h5py.File(io.BytesIO(h5_bytes), "r") as handle:
            features = self.torch.from_numpy(handle["features"][:])
            coords = self.torch.from_numpy(handle["coords"][:])
            patch_size_lv0 = int(handle["coords"].attrs["patch_size_level0"])
        if features.ndim != 3 or features.shape[0] != 1:
            raise ValueError(f"Expected features shaped (1, N, C), got {tuple(features.shape)}")
        if coords.ndim != 3 or coords.shape[0] != 1:
            raise ValueError(f"Expected coords shaped (1, N, 2), got {tuple(coords.shape)}")
        with self.torch.inference_mode():
            embedding = self.model.encode_slide_from_patch_features(
                features.float(), coords.long(), patch_size_lv0
            )
        if not self.torch.isfinite(embedding).all():
            raise RuntimeError("TITAN returned non-finite values")
        return embedding[0].tolist()


class Virchow2Model:
    """Virchow2 tile embedding from a PNG/JPEG: class token + mean patch token."""

    name = "virchow2"
    payload_kind = "image"

    def __init__(self, weights_dir):
        import torch
        import timm
        from timm.layers import SwiGLUPacked

        repo = Path(weights_dir) / "virchow2"
        if not (repo / "config.json").is_file():
            raise FileNotFoundError(f"Virchow2 snapshot not found: {repo}")
        self.torch = torch
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))
        # timm requires an explicit source prefix for a local snapshot directory.
        self.model = timm.create_model(
            f"local-dir:{repo}",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        self.model = self.model.to("cpu").eval()
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        self.preprocess = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )

    @staticmethod
    def warmup_payload():
        return Image.new("RGB", (224, 224))

    def predict(self, image):
        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to("cpu")
        with self.torch.inference_mode():
            output = self.model(tensor)
        class_token = output[:, 0]
        # Tokens 1-4 are register tokens; the card says to ignore them.
        patch_tokens = output[:, 5:]
        embedding = self.torch.cat(
            [class_token, patch_tokens.mean(1)], dim=-1
        )
        if not self.torch.isfinite(embedding).all():
            raise RuntimeError("Virchow2 returned non-finite values")
        return embedding[0].tolist()


def _warmup_h5():
    """Minimal valid patch-feature HDF5 for TITAN's warm-up, as bytes."""
    import h5py
    import numpy as np

    buf = io.BytesIO()
    with h5py.File(buf, "w") as h5:
        # 2x2 grid of distinct non-zero patches: all-zero features produce an
        # all-False background mask (bg_mask = feature_grid != 0), which
        # empties the token sequence; identical coords collapse the grid.
        features = np.arange(4 * 768, dtype="float32").reshape(1, 4, 768) / 1000.0
        h5.create_dataset("features", data=features)
        coords = np.array(
            [[0, 0], [0, 512], [512, 0], [512, 512]], dtype="int64"
        ).reshape(1, 4, 2)
        dataset = h5.create_dataset("coords", data=coords)
        dataset.attrs["patch_size_level0"] = 512

    return buf.getvalue()


LOADERS = {
    "conch": ConchModel,
    "titan": TitanModel,
    "virchow2": Virchow2Model,
}
