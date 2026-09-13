from fastapi import FastAPI, UploadFile, File, Query, Form, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import RTDETR
from PIL import Image
import io

# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="Motorcycle Helmet Detection API",
    description="Helmet detection using RT-DETR",
    version="1.0.0"
)

# --------------------------------------------------
# Load RT-DETR Model
# --------------------------------------------------

model = RTDETR("best.pt")


# --------------------------------------------------
# Helper Function: Read Image
# --------------------------------------------------

async def read_image(file: UploadFile):
    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        return image

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )


# --------------------------------------------------
# Helper Function: Run Detection
# --------------------------------------------------

def get_detections(image):

    results = model(image)

    detections = []

    for box in results[0].boxes:

        class_id = int(box.cls)
        confidence = float(box.conf)

        detections.append({
            "class": results[0].names[class_id],
            "confidence": round(confidence, 3),
            "box": [
                round(x, 1)
                for x in box.xyxy.tolist()[0]
            ]
        })

    return results, detections


# ==================================================
# 1. DETECT API
# ==================================================

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    output: str = Query(
        default="json",
        enum=["json", "image"]
    )
):

    # Read uploaded image
    image = await read_image(file)

    # Run RT-DETR detection
    results, detections = get_detections(image)

    # ------------------------------------------------
    # Return Annotated Image
    # ------------------------------------------------

    if output == "image":

        annotated = results[0].plot()

        # Convert BGR → RGB
        annotated_image = Image.fromarray(
            annotated[:, :, ::-1]
        )

        image_buffer = io.BytesIO()

        annotated_image.save(
            image_buffer,
            format="JPEG"
        )

        image_buffer.seek(0)

        return StreamingResponse(
            image_buffer,
            media_type="image/jpeg"
        )

    # ------------------------------------------------
    # Return JSON
    # ------------------------------------------------

    return {
        "success": True,
        "count": len(detections),
        "detections": detections
    }


# ==================================================
# 2. ASK API
# ==================================================

@app.post("/ask")
async def ask(
    file: UploadFile = File(...),
    question: str = Form(...)
):

    # ------------------------------------------------
    # Read Image
    # ------------------------------------------------

    image = await read_image(file)

    # ------------------------------------------------
    # Run Detection
    # ------------------------------------------------

    results, detections = get_detections(image)

    # ------------------------------------------------
    # Convert Question To Lowercase
    # ------------------------------------------------

    q = question.lower().strip()

    # ------------------------------------------------
    # Keywords Related To Image Detection
    # ------------------------------------------------

    image_keywords = [
        "helmet",
        "driver",
        "person",
        "people",
        "rider",
        "riders",
        "how many",
        "count",
        "wearing",
        "riding",
        "image",
        "photo",
        "no helmet",
        "without helmet"
    ]

    # ------------------------------------------------
    # Step 1: Question Validation
    # ------------------------------------------------

    if not any(
        keyword in q
        for keyword in image_keywords
    ):

        return {
            "success": False,
            "answer":
            "This question does not require image analysis. "
            "Please ask a question related to the detected objects."
        }

    # ------------------------------------------------
    # Step 2: No Detection
    # ------------------------------------------------

    if len(detections) == 0:

        return {
            "success": False,
            "answer":
            "I could not detect any relevant objects "
            "in this image."
        }

    # ------------------------------------------------
    # Step 3: Confidence Check
    # ------------------------------------------------

    avg_confidence = sum(
        detection["confidence"]
        for detection in detections
    ) / len(detections)

    if avg_confidence < 0.40:

        return {
            "success": False,
            "answer":
            f"Objects were detected, but the confidence "
            f"is too low ({avg_confidence:.2f}) "
            f"to give a reliable answer."
        }

    # ------------------------------------------------
    # Step 4: Count Objects
    # ------------------------------------------------

    counts = {}

    for detection in detections:

        class_name = detection["class"]

        counts[class_name] = (
            counts.get(class_name, 0) + 1
        )

    # ------------------------------------------------
    # Step 5: Count Question
    # ------------------------------------------------

    if "how many" in q or "count" in q:

        target = None

        for class_name in counts:

            if class_name.lower() in q:
                target = class_name
                break

        if target:

            return {
                "success": True,
                "answer":
                f"There are {counts[target]} "
                f"{target}(s) detected."
            }

        return {
            "success": True,
            "answer":
            f"Detected objects: {counts}"
        }

    # ------------------------------------------------
    # Step 6: No Helmet Detection
    # ------------------------------------------------

    no_helmet_classes = [
        "no-helmet",
        "no_helmet",
        "without-helmet",
        "without_helmet"
    ]

    no_helmet_count = 0

    for class_name in no_helmet_classes:

        no_helmet_count += counts.get(
            class_name,
            0
        )

    if (
        "no-helmet" in q
        or "no helmet" in q
        or "not wearing" in q
        or "without helmet" in q
        or "without a helmet" in q
    ):

        if no_helmet_count > 0:

            return {
                "success": True,
                "answer":
                f"Yes, {no_helmet_count} "
                f"rider(s) were detected "
                f"without a helmet."
            }

        return {
            "success": True,
            "answer":
            "No riders without a helmet "
            "were detected in this image."
        }

    # ------------------------------------------------
    # Step 7: Helmet Question
    # ------------------------------------------------

    if "helmet" in q:

        helmet_count = 0

        for class_name, count in counts.items():

            if (
                "helmet" in class_name.lower()
                and "no" not in class_name.lower()
                and "without" not in class_name.lower()
            ):
                helmet_count += count

        if helmet_count > 0:

            return {
                "success": True,
                "answer":
                f"{helmet_count} helmet(s) "
                f"were detected."
            }

    # ------------------------------------------------
    # Step 8: Default Answer
    # ------------------------------------------------

    return {
        "success": True,
        "answer":
        f"Based on the image, I detected: {counts}. "
        "Please ask a more specific question."
    }


# ==================================================
# ROOT API
# ==================================================

@app.get("/")
def home():

    return {
        "message":
        "Motorcycle Helmet Detection API is running.",
        "model": "RT-DETR",
        "endpoints": [
            "/detect",
            "/ask"
        ]
    }