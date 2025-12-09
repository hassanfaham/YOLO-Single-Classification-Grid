
"""
Unified training + evaluation script for YOLOv8/YOLOv11 (detection or classification).
- Train with custom hyperparameters, augmentations, and optional layer freezing.
- Evaluate best weights after training.
- Compute precision, recall, F-scores (F1, F2, F0.5), inference speed, and FPS.
- Save all metrics to a text file for experiment tracking.
"""

import os
from pathlib import Path
from ultralytics import YOLO


# ==============================
# USER CONFIGURABLE PARAMETERS
# ==============================
TASK             = "detect"   # "detect" or "classify"
MODEL_PATH       = "yolo11s-cls.pt"
DATA_PATH        = r"path"   # yaml for detection OR folder path for classification
PROJECT          = "./logs"
EXPERIMENT_NAME  = "experiment_01"

EPOCHS           = 60
BATCH_SIZE       = 4
WORKERS          = 2
GPU_ID           = 0

OPTIMIZER        = "Adam"
LR0              = 0.0001
FREEZE_LAYERS    = None

# Augmentations (tuned for detection; YOLO will ignore unsupported args in classification)
# AUGMENT_ARGS = dict(
#     augment=True,
#     hsv_h=0.015,
#     hsv_s=0.7,
#     hsv_v=0.4,
#     translate=0.1,
#     scale=0.5,
#     flipud=0.5,
#     fliplr=0.5,
#     mosaic=0.2,
#     mixup=0.2,
#     copy_paste=0.1,
#     erasing=0.4
# )


def compute_f_scores(metrics: dict):
    """Compute F1, F2, and F0.5 scores from precision/recall."""
    p = metrics.get("metrics/precision(B)", 0)
    r = metrics.get("metrics/recall(B)", 0)

    f1 = (2 * p * r) / (p + r) if (p + r) != 0 else 0
    f2 = (5 * p * r) / (4 * p + r) if (4 * p + r) != 0 else 0
    beta_sq = 0.25
    f05 = ((1 + beta_sq) * p * r) / (beta_sq * p + r) if (beta_sq * p + r) != 0 else 0

    return {"F1_score": f1, "F2_score": f2, "F0.5_score": f05}


def run():
    os.makedirs(PROJECT, exist_ok=True)

    # ==============================
    # MODEL LOADING
    # ==============================
    model = YOLO(MODEL_PATH, task=TASK)

    # ==============================
    # TRAINING
    # ==============================
    model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=640,
        device=GPU_ID,
        workers=WORKERS,
        optimizer=OPTIMIZER,
        lr0=LR0,
        freeze=FREEZE_LAYERS,
        patience=50,
        project=PROJECT,
        name=EXPERIMENT_NAME,
        save=True,
        amp=False,
        plots=True,
        val=False,
        weight_decay=0.00005
        
        # **AUGMENT_ARGS
    )

    # ==============================
    # VALIDATION (using best weights)
    # ==============================
    best_weights = Path(PROJECT) / EXPERIMENT_NAME / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Best weights not found: {best_weights}")

    model = YOLO(str(best_weights))
    results = model.val(
        data=DATA_PATH,
        batch=1,
        device=GPU_ID,
        workers=0,
        verbose=False
    )

    # ==============================
    # METRICS & SPEED
    # ==============================
    metrics = results.results_dict
    metrics.update(compute_f_scores(metrics))

    speed_info = results.speed
    inference_time_ms = speed_info.get("inference", 0)
    metrics["Inference_time_ms"] = inference_time_ms
    metrics["FPS"] = (
        1000.0 / inference_time_ms if isinstance(inference_time_ms, (int, float)) and inference_time_ms > 0 else 0
    )

    # ==============================
    # SAVE METRICS TO FILE
    # ==============================
    log_path = Path(PROJECT) / EXPERIMENT_NAME / "metrics.txt"
    with open(log_path, "a") as f:
        f.write("\n--- New Training Session ---\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    print(f"\n✅ Finished. Metrics saved to: {log_path}")
    print(metrics)


if __name__ == "__main__":
    run()
