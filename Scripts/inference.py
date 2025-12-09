
"""
Unified inference script for YOLOv8/YOLOv11 (detection or classification).
- Accepts a single model (.pt) or a folder of multiple models.
- Accepts a folder of images or a single image file.
- Runs inference, counts OK vs NOK predictions.
- Optional: save annotated results for inspection.
"""

import os
from pathlib import Path
from ultralytics import YOLO
import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)




try:
    import cv2
except ImportError:
    cv2 = None  # only needed if save_viz=True


# ==============================
# USER CONFIGURABLE PARAMETERS
# ==============================
MODEL_PATH   = Path(r"path")  # folder of models OR a single .pt file
IMAGES_PATH  = Path(r"path")  # folder of images OR a single image
CONF         = 0.4
IOU          = 0.7
SAVE_VIZ     = False
OUTPUT_ROOT  = Path("inference_results")


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(path: Path):
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTS:
            yield path
        else:
            raise ValueError(f"Unsupported image format: {path}")
    elif path.is_dir():
        for p in path.iterdir():
            if p.suffix.lower() in IMAGE_EXTS:
                yield p
    else:
        raise FileNotFoundError(f"Invalid images path: {path}")

def run_inference(model_path: Path, images_path: Path):

    model = YOLO(str(model_path))  # task auto-detected from checkpoint
    task = getattr(model, "task", "classify")

    ok_count, nok_count = 0, 0
    out_dir = OUTPUT_ROOT / model_path.stem
    if SAVE_VIZ:
        out_dir.mkdir(parents=True, exist_ok=True)

    for img_file in iter_images(images_path):
        results = model.predict(
            str(img_file),
            conf=CONF,
            iou=IOU,
            save=True,
            verbose=False
        )
        r = results[0]

        names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}

        if task == "detect":
            cls_names = [names[int(b.cls)].lower() for b in (r.boxes or [])]
            if not cls_names or any("nok" in c for c in cls_names):
                nok_count += 1
                pred_label = "NOK"
            else:
                ok_count += 1
                pred_label = "OK"

            if SAVE_VIZ:
                if cv2 is None:
                    raise RuntimeError("OpenCV (cv2) required for save_viz=True")
                img = r.plot()
                cv2.imwrite(str(out_dir / f"{img_file.stem}__{pred_label}.png"), img)

        elif task == "classify":
            if r.probs is None or r.probs.top1 is None:
                top_name, top_score = "bad", 0.0
            else:
                top_idx = int(r.probs.top1)
                top_name = names.get(top_idx, str(top_idx)).lower()
                top_score = float(r.probs.data[top_idx].item())

            if "bad" in top_name.lower(): 
                nok_count += 1
                pred_label = f"NOK_{top_score:.2f}"
                is_ok = False
            else:
                ok_count += 1
                pred_label = f"OK_{top_score:.2f}"
                is_ok = True


            if SAVE_VIZ:
                if cv2 is None:
                    raise RuntimeError("OpenCV (cv2) required for save_viz=True")
                img = r.orig_img.copy()
                color = (0, 255, 0) if is_ok else (0, 0, 255)
                cv2.putText(img, pred_label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, color, 3, cv2.LINE_AA)
                cv2.imwrite(str(out_dir / f"{img_file.stem}__{pred_label}.png"), img)

        else:
            raise ValueError(f"Unsupported YOLO task: {task}")

    print(f"Model: {model_path.stem} → OK: {ok_count} | NOK: {nok_count}")

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model path not found: {MODEL_PATH}")

    if MODEL_PATH.is_file() and MODEL_PATH.suffix.lower() == ".pt":
        model_files = [MODEL_PATH]
    elif MODEL_PATH.is_dir():
        model_files = sorted(MODEL_PATH.glob("*.pt"))
        if not model_files:
            raise FileNotFoundError(f"No .pt models found in {MODEL_PATH}")
    else:
        raise ValueError(f"Invalid model path: {MODEL_PATH}")

    if not IMAGES_PATH.exists():
        raise FileNotFoundError(f"Images path not found: {IMAGES_PATH}")

    for mfile in model_files:
        run_inference(mfile, IMAGES_PATH)


if __name__ == "__main__":
    main()
