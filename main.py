from fastapi import FastAPI, UploadFile, File
from ultralytics import RTDETR
from PIL import Image
import io

app = FastAPI()
model = RTDETR("best.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model(image)

    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": results[0].names[int(box.cls)],
            "confidence": round(float(box.conf), 3),
            "box": [round(x, 1) for x in box.xyxy.tolist()[0]]
        })

    return {"detections": detections, "count": len(detections)}
from fastapi import Form

@app.post("/ask")
async def ask(file: UploadFile = File(...), question: str = Form(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model(image)

    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": results[0].names[int(box.cls)],
            "confidence": float(box.conf)
        })

    q = question.lower()

    # Step 1: Intent routing — does this question even need detection?
    image_keywords = ["helmet", "driver", "person", "how many", "count", "wearing", "riding", "image", "photo"]
    if not any(k in q for k in image_keywords):
        return {"answer": "This question doesn't require image analysis, and I can't answer general questions outside of what's detected in the photo."}

    # Step 2: Confidence guardrail — no detections or all low confidence
    if len(detections) == 0:
        return {"answer": "I could not detect any relevant objects in this image to answer that confidently."}

    avg_conf = sum(d["confidence"] for d in detections) / len(detections)
    if avg_conf < 0.4:
        return {"answer": f"I detected some objects but confidence is too low ({avg_conf:.2f}) to answer reliably."}

    # Step 3: Structured reasoning over detections
    counts = {}
    for d in detections:
        counts[d["class"]] = counts.get(d["class"], 0) + 1

    if "how many" in q or "count" in q:
        target = None
        for cls in counts:
            if cls in q:
                target = cls
        if target:
            return {"answer": f"There are {counts[target]} {target}(s) detected."}
        return {"answer": f"Detected counts: {counts}"}

    if "no-helmet" in q or ("not wearing" in q and "helmet" in q) or ("without" in q and "helmet" in q):
        no_helmet_count = counts.get("no-helmet", 0)
        if no_helmet_count > 0:
            return {"answer": f"Yes, {no_helmet_count} rider(s) detected without a helmet."}
        else:
            return {"answer": "No riders without a helmet were detected in this image."}

    return {"answer": f"Based on detection, I found: {counts}. Please ask a more specific question for a precise answer."}