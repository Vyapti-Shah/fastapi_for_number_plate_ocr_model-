from fastapi import FastAPI, UploadFile, File
import time
import shutil
import os
from ultralytics import YOLO

app = FastAPI()

# Load model once (VERY IMPORTANT)
model = YOLO("model/best.pt")

UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/")
def home():
    return {"message": "Number Plate OCR API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()

    # Save uploaded image temporarily
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run model
    results = model(file_path)

    detections = results[0]

    output = []

    for box in detections.boxes:
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        # For OCR models, usually class names = characters or plate
        label = model.names[cls]

        output.append({
            "text": label,
            "confidence": round(conf, 3)
        })

    end_time = time.time()

    return {
        "predictions": output,
        "total_time": round(end_time - start_time, 3)
    }
