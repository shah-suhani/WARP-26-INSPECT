import os
import json
import shutil
import random
from PIL import Image, ImageFile
import yaml

with open("config.yaml","r") as f:
    config=yaml.safe_load(f)
    
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    raw_img_dir = config['paths']['img_dir']
    json_path = config['paths']['json_path']
    out_dir = config['paths']['yolo_dataset']
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(out_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'labels', split), exist_ok=True)
        
    with open(json_path, 'r') as f:
        jsondata = json.load(f)
        
    damaged_data = []
    classes = set()
    
    if isinstance(jsondata, dict):
        for img_id, img_data in jsondata.items():
            if isinstance(img_data, dict):
                regions = img_data.get('regions', [])
                if len(regions) > 0:
                    filename = img_data.get('filename', img_data.get('name', img_id))
                    damaged_data.append((filename, regions))
                    for region in regions:
                        cls_name = region.get('class')
                        if cls_name:
                            classes.add(cls_name)

    class_list = sorted(list(classes))
    class_map = {cls_name: idx for idx, cls_name in enumerate(class_list)}

    random.seed(42)
    random.shuffle(damaged_data)
    split_idx = int(0.8 * len(damaged_data))
    train_data = damaged_data[:split_idx]
    val_data = damaged_data[split_idx:]
    
    def process_data(data, split_name):
        for filename, regions in data:
            img_path = os.path.join(raw_img_dir, filename)
            if not os.path.exists(img_path):
                continue
                
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception:
                continue
                
            shutil.copy(img_path, os.path.join(out_dir, 'images', split_name, filename))
            
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(out_dir, 'labels', split_name, label_filename)
            
            with open(label_path, 'w') as lf:
                for region in regions:
                    cls_name = region.get('class')
                    if not cls_name or cls_name not in class_map:
                        continue
                        
                    cls_idx = class_map[cls_name]
                    line_data = [str(cls_idx)]
                    
                    if 'all_x' in region and 'all_y' in region:
                        xs = region['all_x']
                        ys = region['all_y']
                        for x, y in zip(xs, ys):
                            norm_x = min(max(x / width, 0.0), 1.0)
                            norm_y = min(max(y / height, 0.0), 1.0)
                            line_data.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])
                            
                    elif 'all_points_x' in region and 'all_points_y' in region:
                        xs = region['all_points_x']
                        ys = region['all_points_y']
                        for x, y in zip(xs, ys):
                            norm_x = min(max(x / width, 0.0), 1.0)
                            norm_y = min(max(y / height, 0.0), 1.0)
                            line_data.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])
                                
                    if len(line_data) > 5:
                        lf.write(" ".join(line_data) + "\n")

    process_data(train_data, 'train')
    process_data(val_data, 'val')
    
    names_str = "\n".join([f"  {idx}: {cls_name}" for cls_name, idx in class_map.items()])
    yaml_content = f"""path: {out_dir}
train: images/train
val: images/val

names:
{names_str}
"""
    with open(os.path.join(out_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

if __name__ == '__main__':
    main()