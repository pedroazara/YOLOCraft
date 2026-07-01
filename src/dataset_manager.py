import random
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtNetwork import QLocalSocket
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "data" / "minecraft_mobs-2"
IMAGES_DIR = DATASET_DIR / "images"
THUMB_WIDTH = 230
GRID_COLUMNS = 3
DETECTOR_SERVER = "yolocraft_detector"


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, image):
    ok, buf = cv2.imencode(Path(path).suffix or ".jpg", image)
    if ok:
        buf.tofile(str(path))
    return ok


def draw_box(image, cx, cy, w, h, label):
    height, width = image.shape[:2]
    x1 = int((cx - w / 2) * width)
    y1 = int((cy - h / 2) * height)
    x2 = int((cx + w / 2) * width)
    y2 = int((cy + h / 2) * height)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image, label, (x1, max(y1 - 5, 12)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
    )
    return image


def cv_to_qpixmap(image_bgr):
    rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class DatasetIndex:
    def __init__(self, dataset_dir):
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.boxes = pd.read_csv(self.dataset_dir / "boxes.csv")
        self.frames = pd.read_csv(self.dataset_dir / "frames.csv")

        merged = self.boxes.merge(self.frames[["frame", "mob"]], on="frame")
        self.id_to_name = (
            merged.drop_duplicates("class_id").set_index("class_id")["mob"].to_dict()
        )
        self.counts = self.boxes.groupby("class_id").size().to_dict()
        self.negative_frames = self.frames.loc[
            self.frames["negative"] == 1, "frame"
        ].tolist()
        self._meta = self.frames.set_index("frame")

    def class_ids(self):
        return sorted(self.id_to_name)

    def name(self, class_id):
        return self.id_to_name[class_id]

    def count(self, class_id):
        return self.counts.get(class_id, 0)

    def sample_frames(self, class_id, n):
        subset = self.boxes[self.boxes["class_id"] == class_id]
        n = min(n, len(subset))
        return subset.sample(n=n) if n else subset.iloc[:0]

    def box_row(self, frame):
        return self.boxes[self.boxes["frame"] == frame].iloc[0]

    def frame_meta(self, frame):
        return self._meta.loc[frame]

    def image_path(self, frame):
        return self.images_dir / f"{frame}.jpg"


class ClickableThumb(QLabel):
    clicked = pyqtSignal(str)

    def __init__(self, frame):
        super().__init__()
        self.frame = frame
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        self.clicked.emit(self.frame)


class DetailDialog(QDialog):
    def __init__(self, index, frame, parent=None):
        super().__init__(parent)
        self.setWindowTitle(frame)
        self.resize(720, 620)
        layout = QVBoxLayout(self)

        row = index.box_row(frame)
        meta = index.frame_meta(frame)
        name = index.name(int(row["class_id"]))

        image = imread_unicode(index.image_path(frame))
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if image is not None:
            draw_box(image, row["cx"], row["cy"], row["w"], row["h"], name)
            pixmap = cv_to_qpixmap(image).scaled(
                680, 460,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            image_label.setPixmap(pixmap)

        info = QTextEdit()
        info.setReadOnly(True)
        info.setMaximumHeight(120)
        info.setText(
            f"frame: {frame}\n"
            f"class: {name} (original id {int(row['class_id'])})\n"
            f"weather: {meta['weather']} | dist: {row['dist_blocks']:.1f} blocks\n"
            f"YOLO label: {int(row['class_id'])} "
            f"{row['cx']:.6f} {row['cy']:.6f} {row['w']:.6f} {row['h']:.6f}"
        )

        layout.addWidget(image_label, stretch=1)
        layout.addWidget(info)


class ExportWorker(QThread):
    progress = pyqtSignal(int, int)
    done = pyqtSignal(str, int)
    failed = pyqtSignal(str)

    def __init__(self, index, selected_ids, out_dir, val_ratio, test_ratio,
                 include_negatives, seed=42):
        super().__init__()
        self.index = index
        self.selected_ids = selected_ids
        self.out_dir = Path(out_dir)
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.include_negatives = include_negatives
        self.seed = seed

    def run(self):
        try:
            remap = {orig: i for i, orig in enumerate(self.selected_ids)}
            names = [self.index.name(orig) for orig in self.selected_ids]

            boxes = self.index.boxes
            positive = boxes[boxes["class_id"].isin(self.selected_ids)]
            items = [(r["frame"], int(r["class_id"]), r["cx"], r["cy"], r["w"], r["h"])
                     for _, r in positive.iterrows()]
            if self.include_negatives:
                items += [(f, None, 0, 0, 0, 0) for f in self.index.negative_frames]

            rng = random.Random(self.seed)
            rng.shuffle(items)

            n_val = int(len(items) * self.val_ratio)
            n_test = int(len(items) * self.test_ratio)
            splits = {
                "val": items[:n_val],
                "test": items[n_val:n_val + n_test],
                "train": items[n_val + n_test:],
            }

            if self.out_dir.exists():
                shutil.rmtree(self.out_dir)

            total = len(items)
            processed = 0
            for split_name, split_items in splits.items():
                img_out = self.out_dir / split_name / "images"
                lbl_out = self.out_dir / split_name / "labels"
                img_out.mkdir(parents=True, exist_ok=True)
                lbl_out.mkdir(parents=True, exist_ok=True)

                for frame, class_id, cx, cy, w, h in split_items:
                    src = self.index.image_path(frame)
                    if src.exists():
                        shutil.copy(src, img_out / src.name)
                        if class_id is None:
                            line = ""
                        else:
                            line = f"{remap[class_id]} {cx} {cy} {w} {h}"
                        (lbl_out / f"{frame}.txt").write_text(line, encoding="utf-8")
                    processed += 1
                    if processed % 100 == 0:
                        self.progress.emit(processed, total)

            yaml_lines = [
                "train: train/images",
                "val: val/images",
                "test: test/images",
                f"nc: {len(names)}",
                "names:",
            ]
            yaml_lines += [f"  {i}: {n}" for i, n in enumerate(names)]
            (self.out_dir / "data.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")

            self.progress.emit(total, total)
            self.done.emit(str(self.out_dir), len(names))
        except Exception as exc:
            self.failed.emit(str(exc))


class DatasetManager(QMainWindow):
    def __init__(self, index):
        super().__init__()
        self.index = index
        self.worker = None
        self.setWindowTitle("YOLOCraft - Dataset Manager")
        self.resize(1240, 780)

        self.tabs = QTabWidget()
        curation = QSplitter(Qt.Orientation.Horizontal)
        curation.addWidget(self._build_class_panel())
        curation.addWidget(self._build_sample_panel())
        curation.setSizes([440, 800])
        self.tabs.addTab(curation, "Curation")
        self.tabs.addTab(self._build_prepared_tab(), "Prepared dataset")
        self.setCentralWidget(self.tabs)

        self._populate_classes()
        self._update_selection_footer()
        self._refresh_prepared_datasets()
        self.statusBar().showMessage(
            f"{len(self.index.class_ids())} classes | "
            f"{len(self.index.boxes)} labeled images | "
            f"{len(self.index.negative_frames)} negatives"
        )

    def _build_class_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter classes...")
        self.search.textChanged.connect(self._filter_classes)

        button_row = QHBoxLayout()
        select_all = QPushButton("Select all")
        select_all.clicked.connect(lambda: self._set_all_checked(True))
        clear_all = QPushButton("Clear all")
        clear_all.clicked.connect(lambda: self._set_all_checked(False))
        button_row.addWidget(select_all)
        button_row.addWidget(clear_all)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Class", "Images"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.cellClicked.connect(self._on_class_clicked)
        self.table.itemChanged.connect(self._on_item_changed)

        self.selection_footer = QLabel()

        layout.addWidget(self.search)
        layout.addLayout(button_row)
        layout.addWidget(self.table, stretch=1)
        layout.addWidget(self.selection_footer)
        layout.addWidget(self._build_export_box())
        return panel

    def _build_export_box(self):
        box = QGroupBox("Export training dataset")
        layout = QVBoxLayout(box)

        ratios = QHBoxLayout()
        ratios.addWidget(QLabel("val"))
        self.val_spin = QDoubleSpinBox()
        self.val_spin.setRange(0.0, 0.5)
        self.val_spin.setSingleStep(0.05)
        self.val_spin.setValue(0.1)
        ratios.addWidget(self.val_spin)
        ratios.addWidget(QLabel("test"))
        self.test_spin = QDoubleSpinBox()
        self.test_spin.setRange(0.0, 0.5)
        self.test_spin.setSingleStep(0.05)
        self.test_spin.setValue(0.1)
        ratios.addWidget(self.test_spin)
        ratios.addStretch()

        self.negatives_check = QCheckBox(
            f"Include negatives ({len(self.index.negative_frames)})"
        )

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output:"))
        self.out_edit = QLineEdit("yolo")
        out_row.addWidget(self.out_edit, stretch=1)

        self.export_btn = QPushButton("Export selected classes")
        self.export_btn.clicked.connect(self.export)
        self.progress = QProgressBar()
        self.progress.setVisible(False)

        layout.addLayout(ratios)
        layout.addWidget(self.negatives_check)
        layout.addLayout(out_row)
        layout.addWidget(self.export_btn)
        layout.addWidget(self.progress)
        return box

    def _build_sample_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        header = QHBoxLayout()
        self.sample_title = QLabel("Select a class to view samples")
        self.sample_title.setStyleSheet("font-weight: bold;")
        header.addWidget(self.sample_title, stretch=1)
        header.addWidget(QLabel("Samples:"))
        self.sample_count = QSpinBox()
        self.sample_count.setRange(1, 24)
        self.sample_count.setValue(9)
        header.addWidget(self.sample_count)
        self.shuffle_btn = QPushButton("Refresh")
        self.shuffle_btn.clicked.connect(self._reload_samples)
        self.shuffle_btn.setEnabled(False)
        header.addWidget(self.shuffle_btn)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.scroll.setWidget(self.grid_host)

        layout.addLayout(header)
        layout.addWidget(self.scroll, stretch=1)

        self.current_class = None
        return panel

    def _populate_classes(self):
        ids = sorted(self.index.class_ids(), key=lambda c: self.index.count(c), reverse=True)
        self.table.blockSignals(True)
        self.table.setRowCount(len(ids))
        for row, class_id in enumerate(ids):
            name_item = QTableWidgetItem(self.index.name(class_id))
            name_item.setData(Qt.ItemDataRole.UserRole, class_id)
            name_item.setFlags(name_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            name_item.setCheckState(Qt.CheckState.Unchecked)
            count_item = QTableWidgetItem(str(self.index.count(class_id)))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, count_item)
        self.table.blockSignals(False)

    def _filter_classes(self, text):
        text = text.lower()
        for row in range(self.table.rowCount()):
            name = self.table.item(row, 0).text().lower()
            self.table.setRowHidden(row, text not in name)

    def _set_all_checked(self, checked):
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            if not self.table.isRowHidden(row):
                self.table.item(row, 0).setCheckState(state)
        self.table.blockSignals(False)
        self._update_selection_footer()

    def _on_item_changed(self, item):
        self._update_selection_footer()

    def _selected_ids(self):
        ids = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item.checkState() == Qt.CheckState.Checked:
                ids.append(item.data(Qt.ItemDataRole.UserRole))
        return sorted(ids)

    def _update_selection_footer(self):
        ids = self._selected_ids()
        total = sum(self.index.count(c) for c in ids)
        self.selection_footer.setText(f"{len(ids)} classes selected | {total} images")

    def _on_class_clicked(self, row, column):
        self.current_class = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        self.shuffle_btn.setEnabled(True)
        self._reload_samples()

    def _reload_samples(self):
        if self.current_class is None:
            return
        name = self.index.name(self.current_class)
        count = self.index.count(self.current_class)
        self.sample_title.setText(f"{name} - {count} images (id {self.current_class})")

        while self.grid.count():
            widget = self.grid.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()

        samples = self.index.sample_frames(self.current_class, self.sample_count.value())
        for i, (_, row) in enumerate(samples.iterrows()):
            frame = row["frame"]
            image = imread_unicode(self.index.image_path(frame))
            thumb = ClickableThumb(frame)
            thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if image is not None:
                draw_box(image, row["cx"], row["cy"], row["w"], row["h"], name)
                pixmap = cv_to_qpixmap(image).scaledToWidth(
                    THUMB_WIDTH, Qt.TransformationMode.SmoothTransformation
                )
                thumb.setPixmap(pixmap)
            thumb.setToolTip(frame)
            thumb.clicked.connect(self._open_detail)
            self.grid.addWidget(thumb, i // GRID_COLUMNS, i % GRID_COLUMNS)

    def _open_detail(self, frame):
        DetailDialog(self.index, frame, self).exec()

    def _build_prepared_tab(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Dataset:"))
        self.prepared_combo = QComboBox()
        self.prepared_combo.currentIndexChanged.connect(self._on_prepared_dataset_changed)
        controls.addWidget(self.prepared_combo, stretch=1)
        controls.addWidget(QLabel("Split:"))
        self.split_combo = QComboBox()
        self.split_combo.currentIndexChanged.connect(self._reload_prepared)
        controls.addWidget(self.split_combo)
        controls.addWidget(QLabel("Samples:"))
        self.prepared_count = QSpinBox()
        self.prepared_count.setRange(1, 48)
        self.prepared_count.setValue(12)
        controls.addWidget(self.prepared_count)
        self.prepared_refresh = QPushButton("Refresh")
        self.prepared_refresh.clicked.connect(self._reload_prepared)
        controls.addWidget(self.prepared_refresh)

        hint = QLabel("Click an image to send it to the detector.")
        hint.setStyleSheet("color: #888;")

        self.prepared_scroll = QScrollArea()
        self.prepared_scroll.setWidgetResizable(True)
        self.prepared_host = QWidget()
        self.prepared_grid = QGridLayout(self.prepared_host)
        self.prepared_scroll.setWidget(self.prepared_host)

        layout.addLayout(controls)
        layout.addWidget(hint)
        layout.addWidget(self.prepared_scroll, stretch=1)

        self.prepared_names = {}
        return panel

    def _refresh_prepared_datasets(self):
        self.prepared_combo.blockSignals(True)
        self.prepared_combo.clear()
        for yaml_path in sorted((ROOT_DIR / "data").glob("**/data.yaml")):
            ds_dir = yaml_path.parent
            self.prepared_combo.addItem(str(ds_dir.relative_to(ROOT_DIR)), str(ds_dir))
        self.prepared_combo.blockSignals(False)
        if self.prepared_combo.count():
            self._on_prepared_dataset_changed()

    def _load_names(self, yaml_path):
        if not yaml_path.exists():
            return {}
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        names = data.get("names", {})
        if isinstance(names, list):
            return {i: n for i, n in enumerate(names)}
        return {int(k): v for k, v in names.items()}

    def _on_prepared_dataset_changed(self):
        ds_dir = self.prepared_combo.currentData()
        if not ds_dir:
            return
        ds_dir = Path(ds_dir)
        self.prepared_names = self._load_names(ds_dir / "data.yaml")
        self.split_combo.blockSignals(True)
        self.split_combo.clear()
        for split in ("train", "val", "test"):
            if (ds_dir / split / "images").exists():
                self.split_combo.addItem(split)
        self.split_combo.blockSignals(False)
        self._reload_prepared()

    def _draw_label_file(self, image, label_path):
        if not label_path.exists():
            return
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cid, cx, cy, w, h = parts
            name = self.prepared_names.get(int(cid), cid)
            draw_box(image, float(cx), float(cy), float(w), float(h), name)

    def _reload_prepared(self):
        while self.prepared_grid.count():
            widget = self.prepared_grid.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()

        ds_dir = self.prepared_combo.currentData()
        split = self.split_combo.currentText()
        if not ds_dir or not split:
            return

        images_dir = Path(ds_dir) / split / "images"
        labels_dir = Path(ds_dir) / split / "labels"
        files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
        if not files:
            return

        sample = random.sample(files, min(self.prepared_count.value(), len(files)))
        for i, img_path in enumerate(sample):
            image = imread_unicode(img_path)
            thumb = ClickableThumb(str(img_path))
            thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if image is not None:
                self._draw_label_file(image, labels_dir / f"{img_path.stem}.txt")
                pixmap = cv_to_qpixmap(image).scaledToWidth(
                    THUMB_WIDTH, Qt.TransformationMode.SmoothTransformation
                )
                thumb.setPixmap(pixmap)
            thumb.setToolTip(img_path.name)
            thumb.clicked.connect(self.send_to_detector)
            self.prepared_grid.addWidget(thumb, i // GRID_COLUMNS, i % GRID_COLUMNS)

    def send_to_detector(self, image_path):
        socket = QLocalSocket()
        socket.connectToServer(DETECTOR_SERVER)
        if socket.waitForConnected(300):
            socket.write(str(image_path).encode("utf-8"))
            socket.flush()
            socket.waitForBytesWritten(500)
            socket.disconnectFromServer()
            self.statusBar().showMessage(f"Sent to detector: {Path(image_path).name}")
            return

        reply = QMessageBox.question(
            self,
            "Detector closed",
            "The detector is not open. Open it with this image?",
        )
        if reply == QMessageBox.StandardButton.Yes:
            subprocess.Popen(
                [sys.executable, "-m", "src.detector_gui", str(image_path)],
                cwd=str(ROOT_DIR),
            )
            self.statusBar().showMessage("Opening detector...")

    def export(self):
        selected = self._selected_ids()
        if not selected:
            QMessageBox.information(self, "No class", "Select at least one class.")
            return

        out_dir = self.index.dataset_dir / self.out_edit.text().strip()
        if out_dir.exists():
            reply = QMessageBox.question(
                self, "Overwrite",
                f"{out_dir} already exists and will be replaced. Continue?",
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self.export_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.statusBar().showMessage("Exporting...")

        self.worker = ExportWorker(
            self.index, selected, out_dir,
            self.val_spin.value(), self.test_spin.value(),
            self.negatives_check.isChecked(),
        )
        self.worker.progress.connect(self._on_export_progress)
        self.worker.done.connect(self._on_export_done)
        self.worker.failed.connect(self._on_export_failed)
        self.worker.start()

    def _on_export_progress(self, processed, total):
        self.progress.setMaximum(total)
        self.progress.setValue(processed)

    def _on_export_done(self, out_dir, num_classes):
        self.export_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.statusBar().showMessage(f"Exported: {num_classes} classes to {out_dir}")
        QMessageBox.information(
            self, "Done",
            f"Dataset generated at:\n{out_dir}\nClasses: {num_classes}",
        )

    def _on_export_failed(self, error):
        self.export_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.statusBar().showMessage("Export failed.")
        QMessageBox.critical(self, "Export error", error)

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(3000)
        super().closeEvent(event)


def main():
    if not (DATASET_DIR / "boxes.csv").exists():
        print(f"Dataset not found at {DATASET_DIR}")
        sys.exit(1)
    app = QApplication(sys.argv)
    window = DatasetManager(DatasetIndex(DATASET_DIR))
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
