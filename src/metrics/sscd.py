import torch
import torch.nn as nn
from torchvision import transforms


class SSCDModelSingleton:
    _instance = None

    @classmethod
    def get_instance(
        cls, model_path="data/ckpts/sscd_disc_large.torchscript.pt", device="cuda"
    ):
        if cls._instance is None:
            cls._instance = torch.jit.load(model_path).to(device)
            cls._instance.eval()
        return cls._instance


def measure_SSCD_similarity(gt_images, images):
    model = SSCDModelSingleton.get_instance()

    ret_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    gt_images = torch.stack([ret_transform(x.convert("RGB")) for x in gt_images]).to(
        "cuda"
    )
    images = torch.stack([ret_transform(x.convert("RGB")) for x in images]).to("cuda")
    with torch.no_grad():
        feat_1 = model(gt_images).clone()
        feat_1 = nn.functional.normalize(feat_1, dim=1, p=2)
        feat_2 = model(images).clone()
        feat_2 = nn.functional.normalize(feat_2, dim=1, p=2)
        return torch.mm(feat_1, feat_2.T)
