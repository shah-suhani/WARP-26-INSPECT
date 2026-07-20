import torch

from app.config import load_config
from app.models.Resnet50.resnet50 import classify
from app.models.Resnet50.resnet50 import load_model as load_resnet
from app.models.YOLO26m.yolo26m import detect, segmented_img
from app.models.YOLO26m.yolo26m import load_model as load_yolo
from app.utils.report import describe, load_vlm

_models = {}


def load_models(config=None):
    global _models
    if _models:
        return _models

    config = config or load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _models = {
        "resnet": load_resnet(config, device),
        "yolo": load_yolo(config),
        "vlm": load_vlm(config),
        "device": device,
    }
    return _models


def analyze(image_path):
    """Run the full pipeline on a single image.

    Returns:
        (segmented_image_or_None, classification_str, damage_text, description_text)
    """
    m = load_models()
    label, confidence = classify(image_path, m["resnet"], m["device"])
    classification = f"{label} ({confidence})"

    if label == "undamaged":
        return None, classification, "No damage detected.", "No damage detected."

    damage_list = detect(image_path, m["yolo"])
    damage_text = (
        "\n".join(f"- {d['class']} (conf: {d['confidence']})" for d in damage_list)
        or "No damage detected."
    )

    seg_img = segmented_img(image_path, m["yolo"])
    description = describe(seg_img, damage_list, m["vlm"])

    return seg_img, classification, damage_text, description


if __name__ == "__main__":
    from app.api.router import create_interface

    demo = create_interface()
    demo.launch(share=True)