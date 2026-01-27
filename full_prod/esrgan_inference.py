import os
# Set environment vars to avoid OpenMP/MKL duplicate-runtime errors on Windows
# (must be set before importing libraries that load OpenMP such as numpy/torch/cv2)
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import time
import cv2
import torch
import importlib
import types

# Provide a small shim for `torchvision.transforms.functional_tensor` when it's
# missing in the installed `torchvision`. basicsr expects this module; some
# torchvision builds place its functions elsewhere. The shim forwards calls to
# `torchvision.transforms.functional` if necessary.
try:
    importlib.import_module('torchvision.transforms.functional_tensor')
except Exception:
    try:
        from torchvision.transforms import functional as _functional

        mod = types.ModuleType('torchvision.transforms.functional_tensor')

        def rgb_to_grayscale(tensor):
            return _functional.rgb_to_grayscale(tensor)

        mod.rgb_to_grayscale = rgb_to_grayscale
        import sys

        sys.modules['torchvision.transforms.functional_tensor'] = mod
    except Exception:
        # If even torchvision.functional isn't available, we'll let the real
        # import error surface later when the functionality is used.
        pass

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


class ESRGANUpscaler:
    def __init__(self, model_path, tile=0, tile_pad=10, pre_pad=0, verbose=True):
        """Create an upscaler instance.

        - Detects device (CUDA if available) and uses fp16 `half` when appropriate.
        - `verbose` toggles printed progress/timing messages.
        """
        self.model_path = model_path
        self.verbose = verbose

        device = "cuda" if torch.cuda.is_available() else "cpu"
        half = torch.cuda.is_available()  # use fp16 on CUDA

        if self.verbose:
            print(f"Initializing ESRGAN upscaler (device={device}, half={half})")

        # RRDBNet architecture for RealESRGAN_x4plus
        self.model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )

        # create the RealESRGANer with explicit device and half settings
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=self.model_path,
            model=self.model,
            tile=tile,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=half,
            device=device,
        )

    def upscale(self, img_path, save_path="static/uploaded/output.png"):
        """Upscale `img_path`, save to `save_path`, and return the path.

        Returns a tuple `(save_path, elapsed_seconds)`.
        """
        if self.verbose:
            print(f"Upscaling {img_path} -> {save_path}")

        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Input image not found: {img_path}")

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        start = time.perf_counter()
        # enhance can take time; measure and optionally print progress
        output, _ = self.upsampler.enhance(img)
        elapsed = time.perf_counter() - start

        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, output_bgr)

        if self.verbose:
            print(f"Upscale finished in {elapsed:.2f}s, saved to {save_path}")

        return save_path, elapsed
