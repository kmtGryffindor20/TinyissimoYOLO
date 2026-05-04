import fiftyone as fo
import os

def export_to_yolo_format(fo_dataset, output_dir, classes):
    """
    Export a FiftyOne dataset to Ultralytics YOLO format.
    Creates images/ and labels/ subdirectories.
    """
    class_to_idx = {c.lower(): i for i, c in enumerate(classes)}

    img_dir = os.path.join(output_dir, "images")
    lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    import shutil
    for sample in fo_dataset.iter_samples(progress=True):
        # Copy image
        fname   = os.path.basename(sample.filepath)
        dst_img = os.path.join(img_dir, fname)
        shutil.copy2(sample.filepath, dst_img)

        # Write label file
        stem    = os.path.splitext(fname)[0]
        dst_lbl = os.path.join(lbl_dir, stem + ".txt")

        dets = getattr(
            getattr(sample, "ground_truth", None), "detections", []) or []

        with open(dst_lbl, "w") as f:
            for det in dets:
                label = (det.label or "").lower()
                if label not in class_to_idx:
                    continue
                x, y, w, h = det.bounding_box   # FiftyOne: [x_tl, y_tl, w, h]
                cx = x + w / 2
                cy = y + h / 2
                cls_idx = class_to_idx[label]
                f.write(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    print(f"Exported {len(fo_dataset)} samples to {output_dir}")


# Usage
import fiftyone.zoo as foz

train = foz.load_zoo_dataset("coco-2017", split="train",
                              label_types=["detections"],
                              classes=["person"])
val   = foz.load_zoo_dataset("coco-2017", split="validation",
                              label_types=["detections"],
                              classes=["person"])

train_view = train.filter_labels("ground_truth",
                  fo.ViewField("label") == "person")
val_view   = val.filter_labels("ground_truth",
                  fo.ViewField("label") == "person")

export_to_yolo_format(train_view, "dataset/train", classes=["person"])
export_to_yolo_format(val_view,   "dataset/val",   classes=["person"])