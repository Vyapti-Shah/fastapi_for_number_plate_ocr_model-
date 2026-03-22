#Step1: Run dir inside the venv (PS D:\uv_Demo> dir)
#Step2: uvicorn main:app --reload (PS D:\uv_Demo> uvicorn main:app --reload)
#Step3: Open the browser and go to http://127.0.0.1:8000/docs (you'll need to add /docs to the URL)
#Step4: Click on Post/predict and then "Try it out" 
#Step5: Upload an image and click on "Execute"
#Step6: The result will be displayed below

from fastapi import FastAPI, UploadFile, File
import time
import shutil
import os
from ultralytics import YOLO

# Initialize FastAPI
app = FastAPI()

# Load YOLO model
model = YOLO("model/best.pt")

# Temporary upload folder
UPLOAD_FOLDER = "temp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Home route
@app.get("/")
def home():
    return {"message": "Number Plate OCR API is running"}


# Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()

    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run YOLO model
    results = model(file_path)
    detections = results[0]

    boxes = detections.boxes

    # Sort boxes left-to-right (important for correct plate order)
    sorted_boxes = sorted(boxes, key=lambda x: float(x.xyxy[0][0]))

    characters = []
    confidences = []

    for box in sorted_boxes:
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        characters.append(label)
        confidences.append(round(conf, 3))

    # Combine characters into full plate number
    plate_text = "".join(characters)

    end_time = time.time()

    return {
        "plate_number": plate_text,
        "characters": characters,
        "confidences": confidences,
        "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0,
        "total_time": round(end_time - start_time, 3)
    }
