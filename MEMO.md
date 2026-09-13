# Project Memo: Motorcycle Helmet Detection API

## 1. Domain & Dataset Choice

I chose motorcycle helmet compliance detection as my domain. This is a practical, real-world safety use case (traffic enforcement, insurance verification, fleet safety monitoring) and satisfies the non-COCO class requirement, since "helmet," "no-helmet," and "driver" (in this specific labeling context) are not standard COCO categories.

**Dataset source:** I used a public dataset from Roboflow Universe ("Motorcycle Helmet" by workspace barry-aprtz, project motorcycle-helmet-wwxcb, version 2, licensed CC BY 4.0, available at https://universe.roboflow.com/barry-aprtz/motorcycle-helmet-wwxcb/dataset/2). The dataset was pre-labeled in YOLO format with 3 classes: `driver`, `helmet`, `no-helmet`.

I did not scrape or self-label additional data due to time constraints, and instead relied on this existing, appropriately licensed, pre-labeled dataset.

## 2. Train/Val/Test Split Strategy

The dataset came pre-split by the original Roboflow contributor into `train/`, `valid/`, and `test/` folders. I used this split as-is rather than re-splitting, since:
- It preserves the original dataset creator's stratification
- It allows any reviewer to cross-check results against the same publicly known split
- Re-splitting risked introducing data leakage between similar/duplicate frames if the source images were extracted from video

## 3. Training Setup (Reproducibility)

- **Model:** RT-DETR (rtdetr-l variant), via Ultralytics library
- **Environment:** Google Colab, Python 3.13, PyTorch 2.11 + CUDA 12.8, Tesla T4 GPU (14913 MiB)
- **Hyperparameters:** epochs=50, image size=640, batch size=default (Ultralytics auto-batch), all other training arguments left at Ultralytics defaults (agnostic_nms=False, amp=True, augment=False)
- **Training command:**
```python
from ultralytics import RTDETR
model = RTDETR("rtdetr-l.pt")
model.train(data="data.yaml", epochs=50, imgsz=640)
```
- **Training time:** Completed within a single Colab session on a T4 GPU.

Exact reproduction steps, including dataset download link and full command sequence, are included in the GitHub repo README.

## 4. Evaluation Metrics

Final validation metrics:

| Metric | Value |
|---|---|
| Precision | 0.88 |
| Recall | 0.56 |
| mAP50 | 0.76 |
| mAP50-95 | 0.41 |

**Per-class mAP50-95:** driver 0.53, helmet 0.36, no-helmet 0.35

**What this tells us:** Precision of 0.88 means that when the model predicts a class, it's correct most of the time — few false positives. Recall of 0.56 means it is missing a substantial number of actual objects in each image — a meaningful limitation for a safety-critical use case, since a missed "no-helmet" detection is the failure mode that matters most in this domain.

**What this does NOT tell us:** mAP alone doesn't reveal *why* recall is low, nor does it capture failure patterns like domain shift (e.g., unusual vehicle types) or class imbalance. Self-reported metrics here are also based on the dataset's own test split, which may not represent the diversity of the hidden evaluation set.

## 5. Failure Case Analysis

1. **Severe class imbalance for "no-helmet":** The validation set contained only 3 instances of "no-helmet" vs. 163 "driver" and 180 "helmet" instances. This directly explains the lower mAP50-95 for "no-helmet" (0.35) — the model simply did not see enough examples of this class to learn it robustly. Root cause: dataset imbalance, not a modeling flaw.

2. **Missed detection on an atypical motorcycle (police bike):** In manual testing, a police-style motorcycle was not detected/boxed correctly by the model. Root cause: likely domain shift — police motorcycles have visually distinct features (light bars, markings, different bodywork) compared to the standard motorcycles that dominate the training data, so the model has not generalized to this visual variation.

3. **Vehicles not detected in some frames:** In several test images, background vehicles (cars, other bikes) were not boxed at all. This is expected behavior, not a bug — the model was only trained on `driver`, `helmet`, `no-helmet` classes, and has no concept of a general "vehicle" class. This highlights a scope limitation worth stating clearly: this model detects helmet-compliance-related entities only, not general traffic objects.

4. **Low recall overall (0.56):** Beyond the specific cases above, the aggregate recall value suggests the model under-detects across classes generally, not just for rare classes. This is consistent with training for only 50 epochs on a moderately sized dataset — likely underfitting given limited training time, rather than an architectural issue with RT-DETR itself.

5. **Confidence-threshold sensitivity (inferred):** Because mAP50 (0.76) is notably higher than mAP50-95 (0.41), the model's bounding boxes are often correctly localized at a loose overlap threshold but less precise at stricter thresholds — suggesting box regression could improve with more training epochs or refined anchor/query settings.

*(Note: due to a tight time budget for this submission, failure cases were identified through a combination of direct visual inspection on test images and analysis of the per-class validation statistics, rather than an exhaustive manual review of all test images.)*

## 6. Part B — Reasoning Layer Design

The `/ask` endpoint implements a hand-written, staged decision process with no agentic framework. Image reading and detection are factored into shared helper functions (`read_image`, `get_detections`) used by both `/detect` and `/ask`, and every response includes a `success` flag alongside the answer text.

**1. Intent routing:** The incoming question is lowercased and checked against a keyword list relevant to the image domain (`helmet`, `driver`, `person`, `rider`, `how many`, `count`, `wearing`, `riding`, `no helmet`, `without helmet`, etc.). If none match, the system immediately returns `success: false` with a message that the question doesn't require image analysis, without proceeding further into the reasoning logic.

**2. Confidence guardrail:** After running detection, if zero objects are found, or if the average detection confidence across all boxes falls below 0.40, the system explicitly returns `success: false` with a message stating it cannot answer confidently, rather than fabricating a response.

**3. Structured reasoning:** For questions that pass both checks, detections are aggregated into per-class counts. "How many"/"count" questions match a target class name against the question text and return its count; "no-helmet"/"without a helmet" phrasing checks the no-helmet count specifically; general "helmet" questions sum all helmet-positive classes; anything else falls through to a default answer listing all detected counts and asking for a more specific question.

**Example of correct "insufficient information" handling:** When asked a question unrelated to the image content (no keyword match, e.g. a general knowledge question) alongside any uploaded image, the intent router responds with `success: false` and a message that the question doesn't require image analysis and it can't answer general questions outside of what's detected in the photo — avoiding any attempt to force an image-based answer to an unrelated question.

## 7. Known Limitations & Honest Disclosure

Given the 5-day (and in practice, a compressed ~10-hour active work) window, the following were intentionally deprioritized:
- No Docker containerization or cloud deployment (bonus items) — the API runs locally via Uvicorn
- No custom data collection/labeling — relied entirely on an existing public dataset
- Training was run for 50 epochs without hyperparameter tuning, which likely leaves accuracy on the table
- The `/ask` reasoning layer uses rule-based logic over detection output rather than an external LLM call for phrasing, to keep the system dependency-light and fully self-contained, with no external API key or billing required; this trades off more natural-sounding answers for reliability, transparency, and reproducibility in how each decision is made
- Detection is run with default RT-DETR confidence/IoU thresholds rather than manually tuned values; in visually dense scenes this can occasionally produce closely spaced or overlapping boxes for adjacent classes (e.g., helmet and driver), which is noted here as an area for future refinement rather than a correctness issue in the underlying detections