import os

from PIL import Image
from ultralytics import YOLO

from app.config import load_config, resolve_path

RUN_NAME = "26m_pro"

def load_model(config):
    yolo_path = resolve_path(config["paths"]["yolo"])
    if not yolo_path.exists():
        raise FileNotFoundError(f"Weights not found at: {yolo_path}")
    return YOLO(str(yolo_path))


def detect(image_path, yolo_model):
    """Run detection and return a list of {'class': str, 'confidence': float} dicts."""
    results = yolo_model(image_path)
    damage_found = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            cls_name = yolo_model.names[cls_id]
            confidence = round(float(box.conf.item()), 2)
            damage_found.append({"class": cls_name, "confidence": confidence})
    return damage_found


def segmented_img(image_path, yolo_model):
    """Run inference and return the annotated/segmented image as a PIL Image."""
    results = yolo_model(image_path, verbose=False)
    annotated = results[0].plot()
    return Image.fromarray(annotated[..., ::-1])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    config = load_config()
    project_dir = resolve_path(config["paths"]["runs_segment"])
    last_weights = os.path.join(str(project_dir), RUN_NAME, "weights", "last.pt")

    if os.path.exists(last_weights):
        model = YOLO(last_weights)
        model.train(resume=True)
        return

    model = YOLO(config["models"]["yolo26m_pt"])
    model.train(
        data=str(resolve_path(config["paths"]["data"])),
        epochs=100,
        imgsz=640,
        batch=32,
        workers=8,
        device=0,
        project=str(project_dir),
        name=RUN_NAME,
        patience=20,
        optimizer="MuSGD",
        amp=True,
        cos_lr=True,
        retina_masks=True,
        overlap_mask=False,
        erasing=0.0,
        auto_augment=False,
        cfg=str(resolve_path(config["paths"]["hyperparameters"])),
    )


if __name__ == "__main__":
    main()