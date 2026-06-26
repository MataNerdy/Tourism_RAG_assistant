"""Image helpers for base64-encoded dataset images."""

from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image


def decode_image(image_base64: str) -> Image.Image:
    """Decode a base64 image string into an RGB PIL image."""

    return Image.open(BytesIO(base64.b64decode(image_base64))).convert("RGB")

