import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def convert_minecraft_mobs2(val_ratio=0.1, test_ratio=0.1, seed=42, include_negatives=True):
    src_dir = ROOT_DIR / "data" / "minecraft_mobs-2"
    images_dir = src_dir / "images"
    out_dir = src_dir / "yolo"

    frame_mob = {}
    with open(src_dir / "frames.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame_mob[row["frame"]] = row["mob"]

    boxes_by_frame = defaultdict(list)
    class_names = {}
    with open(src_dir / "boxes.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame = row["frame"]
            class_id = int(row["class_id"])
            boxes_by_frame[frame].append((class_id, row["cx"], row["cy"], row["w"], row["h"]))
            class_names[class_id] = frame_mob.get(frame, "")

    names = [class_names[i] for i in sorted(class_names)]

    frames = list(frame_mob.keys())
    if not include_negatives:
        frames = [f for f in frames if f in boxes_by_frame]
    frames = [f for f in frames if (images_dir / f"{f}.jpg").exists()]

    random.seed(seed)
    random.shuffle(frames)

    n_val = int(len(frames) * val_ratio)
    n_test = int(len(frames) * test_ratio)
    splits = {
        "val": frames[:n_val],
        "test": frames[n_val:n_val + n_test],
        "train": frames[n_val + n_test:],
    }

    for split_name, split_frames in splits.items():
        img_out = out_dir / split_name / "images"
        lbl_out = out_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for frame in split_frames:
            src_img = images_dir / f"{frame}.jpg"
            shutil.copy(src_img, img_out / src_img.name)

            lines = [f"{cid} {cx} {cy} {w} {h}" for cid, cx, cy, w, h in boxes_by_frame.get(frame, [])]
            (lbl_out / f"{frame}.txt").write_text("\n".join(lines), encoding="utf-8")

    yaml_path = out_dir / "data.yaml"
    yaml_lines = [
        f"train: train/images",
        f"val: val/images",
        f"test: test/images",
        f"nc: {len(names)}",
        "names:",
    ]
    yaml_lines += [f"  {i}: {name}" for i, name in enumerate(names)]
    yaml_path.write_text("\n".join(yaml_lines), encoding="utf-8")

    return out_dir


if __name__ == "__main__":
    out = convert_minecraft_mobs2()
    print(f"dataset generated at: {out}")
