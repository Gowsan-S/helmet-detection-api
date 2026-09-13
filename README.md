# Motorcycle Helmet Detection API

An object detection API using RT-DETR, fine-tuned to detect `driver`, `helmet`, and `no-helmet` on motorcycle riders, with a natural-language reasoning layer built on top.

## Setup

1. Clone this repo:
```bash
git clone https://github.com/Gowsan-S/helmet-detection-api.git
cd helmet-detection-api
```

2. Install dependencies:
```bash
python -m pip install -r requirements.txt
```

3. Make sure `best.pt` (the trained model weights) is in the same folder as `main.py`.

## Running the API

```bash
python -m uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.
Interactive docs (test all endpoints directly in browser): `http://127.0.0.1:8000/docs`

## Endpoint 0: `/` (Health/Status)

Returns basic API status and lists available endpoints.

**Request:** `GET /`

**Sample response:**
```json
{
  "message": "Motorcycle Helmet Detection API is running.",
  "model": "RT-DETR",
  "endpoints": ["/detect", "/ask"]
}
```

## Endpoint 1: `/detect` (Part A)

Accepts an image, returns detected objects with bounding boxes and confidence scores. Can optionally return an annotated image instead of JSON.

**Request:** `POST /detect`, form-data with key `file` = image file
**Optional query parameter:** `output` = `json` (default) or `image`

**Sample response (`output=json`, default):**
```json
{
  "success": true,
  "count": 2,
  "detections": [
    {"class": "driver", "confidence": 0.91, "box": [120.5, 85.3, 310.2, 400.8]},
    {"class": "helmet", "confidence": 0.87, "box": [140.1, 60.2, 210.4, 130.9]}
  ]
}
```

**With `output=image`:** returns a JPEG image with bounding boxes, class labels, and confidence scores drawn directly on it, instead of JSON.

## Endpoint 2: `/ask` (Part B)

Accepts an image and a natural-language question, reasons over the detection output, and returns a plain-language answer.

**Request:** `POST /ask`, form-data with keys `file` = image file, `question` = text

**Sample requests and responses:**

Question: `"is anyone riding without a helmet?"`
```json
{"success": true, "answer": "Yes, 1 rider(s) were detected without a helmet."}
```

Question: `"how many drivers are there?"`
```json
{"success": true, "answer": "There are 2 driver(s) detected."}
```

Question: `"what is the capital of France?"`
```json
{"success": false, "answer": "This question does not require image analysis. Please ask a question related to the detected objects."}
```

## Model Details

- **Architecture:** RT-DETR (rtdetr-l), via Ultralytics
- **Classes:** driver, helmet, no-helmet
- **Dataset:** [Motorcycle Helmet Detection](https://universe.roboflow.com/barry-aprtz/motorcycle-helmet-wwxcb/dataset/2), Roboflow Universe, CC BY 4.0
- **Training:** 50 epochs, image size 640, Google Colab (Tesla T4 GPU)
- **Metrics:** mAP50 = 0.76, mAP50-95 = 0.41, Precision = 0.88, Recall = 0.56

See `MEMO.md` for full dataset sourcing rationale, evaluation discussion, and failure case analysis.