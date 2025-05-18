import sys
import json
import requests
from PIL import Image
from io import BytesIO
from ultralytics import YOLO

def download_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    path = "/tmp/input.jpg"
    img.save(path)
    return path

def predict(image_url):
    image_path = download_image(image_url)
    model = YOLO("yolov8n.pt")
    results = model(image_path)[0]
    
    output = []
    for box in results.boxes:
        x, y, w, h = map(int, box.xywh[0])
        label_idx = int(box.cls[0])
        label = results.names[label_idx]
        confidence = float(box.conf[0])
        output.append({
            "x": x, "y": y, "w": w, "h": h,
            "label": label,
            "confidence": confidence
        })
    return output

if __name__ == "__main__":
    image_url = sys.argv[1]
    result = predict(image_url)
    print(json.dumps(result))
