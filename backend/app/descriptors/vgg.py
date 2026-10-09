import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
THUMB = 64  # MIRFLICKR thumbnails are 64x64


def pil_to_uint8(img: Image.Image) -> torch.Tensor:
    """PIL image -> uint8 tensor (3, 64, 64). Anything not 64x64 (e.g. a query
    uploaded by the user) is first brought down to 64x64 so it looks like the
    database thumbnails."""
    img = img.convert("RGB")
    if img.size != (THUMB, THUMB):
        img = img.resize((THUMB, THUMB), Image.BILINEAR)
    return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)  # uint8


def preprocess(x_uint8: torch.Tensor) -> torch.Tensor:
    """(N,3,64,64) uint8 -> (N,3,224,224) float, ImageNet-normalized.
    Works on CPU or GPU (resize is done on the tensor's device)."""
    x = x_uint8.float() / 255.0
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def build_extractor(features_path=None, device="cpu"):
    """VGG16 conv part + global average pooling -> 512-d vector.
    features_path: saved weights file (no internet needed). If None, the
    ImageNet weights are downloaded by torchvision."""
    if features_path is None:
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = model.features
    else:
        features = models.vgg16(weights=None).features
        features.load_state_dict(torch.load(features_path, map_location="cpu"))
    extractor = torch.nn.Sequential(features,
                                    torch.nn.AdaptiveAvgPool2d(1),
                                    torch.nn.Flatten())
    return extractor.to(device).eval()


@torch.no_grad()
def embed(extractor, x_uint8: torch.Tensor, device="cpu", fp16=False) -> np.ndarray:
    """(N,3,64,64) uint8 -> (N,512) float32 numpy, L2-normalized."""
    x = preprocess(x_uint8.to(device, non_blocking=True))
    if fp16 and device == "cuda":
        with torch.autocast("cuda", dtype=torch.float16):
            f = extractor(x)
    else:
        f = extractor(x)
    f = F.normalize(f.float(), dim=1)
    return f.cpu().numpy()
