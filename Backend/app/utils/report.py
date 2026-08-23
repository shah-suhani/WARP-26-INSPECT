import base64
import os
from io import BytesIO

from openai import OpenAI

from app.config import load_config

SYSTEM_PROMPT = (
    "You are an expert car damage inspector working for an insurance company. "
    "Your job is to assess vehicle damage from images, identify damage types and "
    "locations, estimate repair complexity and cost range, and determine if the "
    "claim is valid. Be precise and professional."
)

DEFAULT_MODEL = "Qwen/Qwen3.5-397B-A17B:novita"


def load_vlm(config=None):
    """Build the HF-router client. HF_TOKEN wins; config.yaml is the fallback."""
    api_key = os.getenv("HF_TOKEN")

    if not api_key:
        config = config if config is not None else load_config()
        api_key = (config.get("api") or {}).get("qwen")

    if not api_key or api_key.startswith("YOUR_"):
        raise ValueError(
            "No Hugging Face router key found. Set HF_TOKEN in Backend/.env "
            "(copy .env.example), or fill api.qwen in Backend/config.yaml."
        )

    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=api_key,
    )


def describe(seg_img, damage_list, vlm, model: str = DEFAULT_MODEL) -> str:
    """Generate a written damage report from the segmented image and detected classes."""
    img_data = _encode_image(seg_img)
    damage_classes = [d["class"] for d in damage_list]

    response = vlm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"The following damage types were detected: {damage_classes}. "
                            "Analyse the segmented image and provide: Damage location and "
                            "type, Severity, Estimated repair complexity, a summary of the "
                            "damages."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_data}"},
                    },
                ],
            },
        ],
    )
    return response.choices[0].message.content


def _encode_image(image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")