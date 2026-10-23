"""HTTP contract checks with fake models; these do not validate model accuracy."""
import base64
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from fastapi.testclient import TestClient
from PIL import Image

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "openshift-ai"))
spec = importlib.util.spec_from_file_location("pathology_server", root / "openshift-ai/server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class FakeImageModel:
    name = "conch"
    payload_kind = "image"

    @staticmethod
    def warmup_payload():
        return Image.new("RGB", (8, 8))

    def predict(self, image):
        return [float(image.width), float(image.height)]


class FakeH5Model(FakeImageModel):
    name = "titan"
    payload_kind = "h5"

    @staticmethod
    def warmup_payload():
        import h5py
        import numpy as np

        buf = io.BytesIO()
        with h5py.File(buf, "w") as h5:
            h5.create_dataset("features", data=np.zeros((1, 1, 8), dtype="float32"))
            h5.create_dataset("coords", data=np.zeros((1, 1, 2), dtype="int64"))
            h5["coords"].attrs["patch_size_level0"] = 512
        return buf.getvalue()

    def predict(self, h5_bytes):
        # Mirror the real TitanModel contract: parse the h5 and reject wrong
        # dataset shapes so the 400-on-bad-shape path is exercised.
        import h5py

        with h5py.File(io.BytesIO(h5_bytes), "r") as handle:
            features = handle["features"].shape
            coords = handle["coords"].shape
            handle["coords"].attrs["patch_size_level0"]
        if len(features) != 3 or features[0] != 1:
            raise ValueError(f"Expected features shaped (1, N, C), got {features}")
        if len(coords) != 3 or coords[0] != 1:
            raise ValueError(f"Expected coords shaped (1, N, 2), got {coords}")
        return [float(features[1])]


def build_app(monkeypatch_loader):
    return server.create_app(monkeypatch_loader)


def h5_payload():
    """Minimal HDF5 with features, coords, and the patch-size attribute."""
    import h5py
    import numpy as np

    handle = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    with h5py.File(handle.name, "w") as h5:
        h5.create_dataset("features", data=np.zeros((1, 2, 8), dtype="float32"))
        coords = h5.create_dataset("coords", data=np.zeros((1, 2, 2), dtype="int64"))
        coords.attrs["patch_size_level0"] = 512
    handle.close()
    return handle.name


class ServerTests(unittest.TestCase):
    def client_for(self, loader):
        self.context = TestClient(server.create_app(loader))
        return self.context.__enter__()

    def tearDown(self):
        if not hasattr(self, "context"):
            return
        self.context.__exit__(None, None, None)

    def payload(self, raw):
        return {"instances": [{"image": {"b64": base64.b64encode(raw).decode()}}]}

    def png(self):
        output = io.BytesIO()
        Image.new("RGB", (32, 24)).save(output, format="PNG")
        return output.getvalue()

    def test_png_readiness_and_routing(self):
        client = self.client_for(lambda: FakeImageModel())
        self.assertEqual(client.get("/health/ready").status_code, 200)
        response = client.post("/v1/models/conch:predict", json=self.payload(self.png()))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predictions": [[32.0, 24.0]]})


    def test_h5_path(self):
        path = h5_payload()
        try:
            client = self.client_for(lambda: FakeH5Model())
            with open(path, "rb") as handle:
                raw = handle.read()
            response = client.post("/v1/models/titan:predict", json=self.payload(raw))
            self.assertEqual(response.status_code, 200)
            # h5_payload uses features (1, 2, 8): FakeH5Model returns N=2.
            self.assertEqual(response.json(), {"predictions": [[2.0]]})
        finally:
            import os
            os.unlink(path)

    def test_bad_payload_rejected_on_titan(self):
        """Non-HDF5 bytes fail in predict; bare-bones server returns 400."""
        client = self.client_for(lambda: FakeH5Model())
        response = client.post("/v1/models/titan:predict", json=self.payload(b"not hdf5"))
        self.assertEqual(response.status_code, 400)


    def test_invalid_images_and_payloads(self):
        client = self.client_for(lambda: FakeImageModel())
        dumps = json.dumps
        for payload in [None, {}, {"instances": []}, self.payload(b"not an image"),
                        {"instances": [{"image": {"b64": "!!!"}}]}]:
            response = client.post("/v1/models/conch:predict", content=dumps(payload))
            self.assertEqual(response.status_code, 400)


    def test_bad_model_name_prevents_startup(self):
        import os
        old = os.environ.get("MODEL_NAME")
        os.environ["MODEL_NAME"] = "resnet"
        try:
            with self.assertRaises(SystemExit):
                server.create_app()
        finally:
            if old is None:
                del os.environ["MODEL_NAME"]
            else:
                os.environ["MODEL_NAME"] = old

    def test_bad_weights_prevent_startup(self):
        def fail():
            raise FileNotFoundError("missing checkpoint")
        with self.assertRaises(FileNotFoundError):
            with TestClient(server.create_app(fail)):
                pass


if __name__ == "__main__":
    unittest.main()
