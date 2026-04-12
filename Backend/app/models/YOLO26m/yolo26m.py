import os
from ultralytics import YOLO
import yaml

with open("config.yaml","r") as f:
    config= yaml.safe_load(f)


def main():
    project_dir = config['paths']['runs_segment']
    run_name = '26m_pro'
    last_weights = os.path.join(project_dir, run_name, 'weights', 'last.pt')
    
    if os.path.exists(last_weights):
        model = YOLO(last_weights)
        model.train(resume=True)
    else:
        model = YOLO(config['models']['yolo26m_pt'])
        
        model.train(
            data = config['paths']['data'],
            epochs = 100,
            imgsz = 640,
            batch = 32,
            workers = 8,
            device = 0,
            project = project_dir,
            name = run_name,
            patience = 20,
            optimizer = 'MuSGD',
            amp = True,
            cos_lr = True,
            retina_masks = True,
            overlap_mask = False,
            erasing = 0.0,
            auto_augment = False,
            cfg = config['paths']['hyperparameters']
        )

if __name__ == '__main__':
    main()