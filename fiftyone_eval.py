print("Loading modules...")
import custom_modules
import fiftyone as fo
import fiftyone.zoo as foz

import fiftyone as fo
import torch
from ultralytics import YOLO


model = YOLO("D:\\VideoSummarizer\\TinyissimoYOLO\\runs\\tinyissimoYOLO\\weights\\best.pt")
# model = YOLO("yolov8n.pt")  # load a pretrained model (or your custom model)
print("Model loaded successfully.")

dataset = foz.load_zoo_dataset(
    "coco-2017", split="validation", classes=["person"], seed=51
)
print(f"Dataset loaded with {len(dataset)} samples.")
dataset.persistent = True


# Keep only person detections in the label field
person_view = dataset.filter_labels(
    "ground_truth",
    fo.ViewField("label") == "person",
).filter_labels(
    "ground_truth",
    fo.ViewField("bounding_box")[2] > (12/128)
).filter_labels(
    "ground_truth",
    fo.ViewField("bounding_box")[3] > (12/128)
)
# Run it

person_view.apply_model(model, "predictions")


results = person_view.evaluate_detections(
    "predictions",  # your model's predictions field
    gt_field="ground_truth",
    eval_key="eval",
    iou=0.50,
    method="coco",  # or "open-images", "voc"
    classwise=True,
    compute_mAP=True,
)

# Print report
results.print_report(classes=["person"])

# mAP
print(f"mAP@50: {results.mAP():.4f}")


session = fo.launch_app(person_view)
session.wait()
