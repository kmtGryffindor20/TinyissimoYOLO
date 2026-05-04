import os
from pathlib import Path
from tqdm import tqdm

def filter_small_boxes(
    labels_dir: str,
    min_area:   float = 0.001,   # fraction of image area
    min_w:      float = 0.02,    # fraction of image width
    min_h:      float = 0.02,    # fraction of image height
    dry_run:    bool  = True,    # set False to actually write files
) -> dict:
    """
    Filter small boxes from Ultralytics YOLO label files in-place.

    Label format (one box per line):
        class_idx cx cy w h   (all normalised to [0,1])

    Args
    ----
    labels_dir : path to the labels/ directory
    min_area   : minimum w×h as fraction of image. Default 0.001
    min_w      : minimum width  as fraction of image. Default 0.02
    min_h      : minimum height as fraction of image. Default 0.02
    dry_run    : if True, only reports what would be removed (safe to run first)

    Returns
    -------
    dict with 'total', 'removed', 'files_modified', 'files_emptied'
    """
    stats = {"total": 0, "removed": 0, "files_modified": 0, "files_emptied": 0}

    label_files = list(Path(labels_dir).rglob("*.txt"))
    print(f"Found {len(label_files)} label files in {labels_dir}")

    for lf in tqdm(label_files):
        lines = lf.read_text().strip().splitlines()
        kept  = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                kept.append(line)   # malformed line — keep and skip
                continue

            cls, cx, cy, w, h = parts
            w, h = float(w), float(h)
            stats["total"] += 1

            if w * h < min_area or w < min_w or h < min_h:
                stats["removed"] += 1
                continue   # drop this box

            kept.append(line)

        # Only write if something changed
        if len(kept) != len(lines):
            stats["files_modified"] += 1
            if len(kept) == 0:
                stats["files_emptied"] += 1

            if not dry_run:
                lf.write_text("\n".join(kept))

    action = "Would remove" if dry_run else "Removed"
    print(f"{action} {stats['removed']}/{stats['total']} boxes "
          f"across {stats['files_modified']} files "
          f"({stats['files_emptied']} files emptied)")
    return stats


# ── Usage ────────────────────────────────────────────────────────────

# Step 1: dry run first to see what would be removed
stats = filter_small_boxes(
    "../datasets/coco-2017/val/labels",
    min_area = 0.09375 * 0.09375,  # 1/16 area (for 256×256 images)
    min_w    = 0.09375,           # 1/16 width
    min_h    = 0.09375,           # 1/16 height
    dry_run  = False,    # safe — no files changed
)

stats_ = filter_small_boxes(
    "../datasets/coco-2017/train/labels",
    min_area = 0.09375 * 0.09375,  # 1/16 area (for 256×256 images)
    min_w    = 0.09375,           # 1/16 width
    min_h    = 0.09375,           # 1/16 height
    dry_run  = False,   # actually remove small boxes
)



