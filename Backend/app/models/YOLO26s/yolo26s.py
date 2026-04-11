import os
from ultralytics import YOLO

def main():
    project_dir = '/home/warp_inspect_new/project/runs/segment'
    run_name = '26s_02'
    last_weights = os.path.join(project_dir, run_name, 'weights', 'last.pt')
    
    if os.path.exists(last_weights):
        model = YOLO(last_weights)
        model.train(resume=True)
    else:
        model = YOLO('/home/warp_inspect_new/project/yolo/yolo26s-seg.pt')
        
        model.train(

            data='/home/warp_inspect_new/project/yolo_dataset/data.yaml',
            epochs = 150,
            imgsz = 1280,
            batch = 16,
            workers = 8,
            device = 0,

            project = project_dir,
            name = run_name,
            patience = 30,
            save = True,
            save_period = 10,
            optimizer = 'MuSGD',
            amp = False,

            degrees=5.0,
            fliplr=0.5,
        )

if __name__ == '__main__':
    main()