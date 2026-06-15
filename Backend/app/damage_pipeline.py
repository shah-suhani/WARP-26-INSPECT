import os
import torch
import torch.nn as nn
import base64
import gradio as gr
from PIL import Image
from openai import OpenAI
from torchvision import models, transforms
from io import BytesIO
from ultralytics import YOLO
import httpx
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
config_path = BASE_DIR / "config.yaml"

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

yolo_path = BASE_DIR / config['paths']['yolo']
resnet_path = BASE_DIR / config['paths']['resnet']
_models={}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_models():
    global _models
    if _models:
        return _models
    
    resnet = models.resnet50(weights='DEFAULT')  
    resnet.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(resnet.fc.in_features, 2)  
    )
    resnet.load_state_dict(torch.load(resnet_path, map_location=device))
    resnet.to(device).eval()
    yolo=YOLO(yolo_path)

    vlm=OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key = BASE_DIR / config['api']['qwen'],
        http_client=httpx.Client(verify=False)
    )

    _models.update({'resnet': resnet, 'yolo': yolo,
                    'vlm': vlm, 'device': device})
    return _models

test_transforms=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def classify(image_path,model,device,transform):
    try:
        img=Image.open(image_path).convert('RGB')
    except:
        print("error in opening")
    
    tensor=transform(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        output=model(tensor)
        prob=torch.softmax(output,dim=1)
        confidence,pred=torch.max(prob,1)
    
    classes = ['undamaged', 'damaged']
    label = classes[pred.item()]

    confidence=confidence.item() *100
    return label,confidence

def detect(image_path,yolo_model):
    results=yolo_model(image_path)
    damage_found=[]
    for result in results:
        for box in result.boxes:
            cls_id=int(box.cls.item())
            cls_name=yolo_model.names[cls_id]
            confidence=round(float(box.conf.item()),2)
            damage_found.append({'class':cls_name,
            'confidence':confidence})
    return damage_found

def segmented_img(image_path,yolo_model):
    results=yolo_model(image_path,verbose=False)
    annotated=results[0].plot()
    return Image.fromarray(annotated[..., ::-1])

def describe(seg_img,damage_list,vlm):
    buf=BytesIO()
    seg_img.save(buf,format='JPEG')
    buf.seek(0)
    img_data=base64.b64encode(buf.getvalue()).decode('utf-8')

    damage_classes=[d['class']for d in damage_list]
    description=vlm.chat.completions.create(
        model="Qwen/Qwen3.5-397B-A17B:novita",
        messages=[
            {
                "role": "system",
                "content": "You are an expert car damage inspector working for an insurance company. Your job is to assess vehicle damage from images, identify damage types and locations, estimate repair complexity and cost range, and determine if the claim is valid. Be precise and professional."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text",
                     "text": f"The following damage types were detected: {damage_classes}. Analyse the segmented image and provide: Damage location and type , Severity  , Estimated repair complexity , a summary of the damages."},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}}
                ]
            }
        ]
    )
    return description.choices[0].message.content


def analyze(image_path):
    m=load_models()
    label, conf = classify(image_path, m['resnet'], m['device'],test_transforms)
    classification=f"{label} ({conf})"

    if label=='undamaged':
        return None,classification,"NO damage detected","no damage detected"

    damage_list=detect(image_path,m['yolo'])
    damage_text = "\n".join([f"- {d['class']} (conf: {d['confidence']})"
                              for d in damage_list]) or "No damage detected."

    seg_img=segmented_img(image_path,m['yolo'])
    description=describe(seg_img,damage_list,m['vlm'])

    return seg_img,classification,damage_text,description


demo=gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="filepath", label="Upload car image"),
    outputs=[
        gr.Image(type="pil",label="segmented_image"),
        gr.Textbox(label="Classification"),
        gr.Textbox(label="Detected_damage"),
        gr.Textbox(label="damage summary",lines=8),
    ],
    title="car damage inspector"
)

if __name__ =="__main__":
    load_models()
    demo.launch(share=True)
