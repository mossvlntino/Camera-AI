from ultralytics import YOLO

model = YOLO("yolov8n.pt")  
model.train(data=r"C:\Users\User\Documents\Tugas Amos\VSC\camera_ai\roboflow_dataset/data.yaml", epochs=200, imgsz=675, batch=16, augment=True)