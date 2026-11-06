from ultralytics import YOLO

model = YOLO("yolov8n.pt")

print("Model classes:")
print(model.names)