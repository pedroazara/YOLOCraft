def draw_yolo_boxes(img, label_path, class_names):

    h, w = img.shape[:2]

    with open(label_path) as f:
        lines = f.readlines()

    for line in lines:

        cls, xc, yc, bw, bh = map(float, line.split())

        cls = int(cls)

        x1 = int((xc - bw/2) * w)
        y1 = int((yc - bh/2) * h)

        x2 = int((xc + bw/2) * w)
        y2 = int((yc + bh/2) * h)

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            class_names[cls],
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return img