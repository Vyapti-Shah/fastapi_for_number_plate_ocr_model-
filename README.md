# Number Plate OCR Model

This repository contains a FastAPI-based OCR API for recognizing number plate characters from images using an Ultralytics YOLO model.

## What it does

- Loads a YOLO model from `model/best.pt`
- Exposes a web API via FastAPI
- Accepts an image upload at `/predict`
- Runs detection on the image and returns:
  - `plate_number`: detected plate characters concatenated in reading order
  - `characters`: list of individual detected labels
  - `confidences`: confidence score for each detected character
  - `avg_confidence`: average confidence
  - `total_time`: total processing time in seconds

## Files

- `main.py` - FastAPI application and prediction endpoint
- `req.txt` - required Python packages for this project
- `model/best.pt` - trained YOLO model file (must exist for inference)
- `temp/` - temporary folder used to store uploaded images
- `stepstorun.txt` - basic run instructions

## Prerequisites

- Python 3.10+ installed
- `model/best.pt` must exist in the `model/` folder
- Windows PowerShell or any terminal with access to the project folder

## Setup

1. Open a terminal in the project root directory:

```powershell
cd D:\AIML\NumberPlate_OCR_Model
```

2. (Optional) Create a virtual environment:

```powershell
py -m venv .venv
```

3. Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
pip install -r req.txt
```

## Run the application

Start the API server with:

```powershell
uvicorn main:app --reload
```

Then open the API docs in a browser:

```text
http://127.0.0.1:8000/docs
```

## Using the API

1. In the Swagger UI, choose `POST /predict`
2. Click `Try it out`
3. Upload an image file containing a number plate
4. Click `Execute`
5. View the returned JSON result

## Notes

- The app saves uploaded images in the `temp/` directory.
- If the model file `model/best.pt` is missing, the app will fail to start.
- If you want to use `requirements.txt` instead of `req.txt`, copy or rename `req.txt`.

## Example response

```json
{
  "plate_number": "ABC1234",
  "characters": ["A", "B", "C", "1", "2", "3", "4"],
  "confidences": [0.987, 0.952, 0.941, 0.889, 0.895, 0.912, 0.873],
  "avg_confidence": 0.917,
  "total_time": 0.451
}
```

## Troubleshooting

- If `uvicorn` is not found, make sure the virtual environment is activated and dependencies are installed.
- If the model fails to load, verify that `model/best.pt` exists and is compatible with the installed `ultralytics` version.
- If uploads fail, confirm `temp/` is writable by the current user.
