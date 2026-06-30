import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def imread_unicode(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, image):
    ext = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if ok:
        buf.tofile(str(path))
    return ok


def cv_to_qpixmap(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def discover_models(root):
    patterns = [
        "pretrained_models/*.pt",
        "models/**/*.pt",
        "notebooks/**/weights/*.pt",
        "*.pt",
    ]
    found = []
    seen = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path not in seen:
                seen.add(path)
                found.append(path)
    return found


class ImageView(QLabel):
    file_dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(560, 420)
        self.setStyleSheet("border: 1px solid #555; border-radius: 4px;")
        self._pixmap = None
        self._placeholder = "Arraste uma imagem aqui ou clique em 'Abrir imagem'"
        self.setText(self._placeholder)

    def set_image(self, pixmap):
        self._pixmap = pixmap
        self._rescale()

    def clear_image(self):
        self._pixmap = None
        self.setText(self._placeholder)

    def resizeEvent(self, event):
        self._rescale()
        super().resizeEvent(event)

    def _rescale(self):
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(IMAGE_EXTENSIONS):
                self.file_dropped.emit(path)
                return


class ModelLoader(QThread):
    loaded = pyqtSignal(object, dict)
    failed = pyqtSignal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            model = YOLO(self.path)
            self.loaded.emit(model, dict(model.names))
        except Exception as exc:
            self.failed.emit(str(exc))


class InferenceWorker(QThread):
    done = pyqtSignal(object, list, float)
    failed = pyqtSignal(str)

    def __init__(self, model, image_bgr, conf, device):
        super().__init__()
        self.model = model
        self.image_bgr = image_bgr
        self.conf = conf
        self.device = device

    def run(self):
        try:
            start = time.time()
            result = self.model.predict(
                source=self.image_bgr,
                conf=self.conf,
                device=self.device,
                verbose=False,
            )[0]
            elapsed = time.time() - start

            annotated = np.ascontiguousarray(result.plot())
            detections = []
            for box in result.boxes:
                cls = int(box.cls[0])
                detections.append((result.names[cls], float(box.conf[0])))
            detections.sort(key=lambda d: d[1], reverse=True)

            self.done.emit(annotated, detections, elapsed)
        except Exception as exc:
            self.failed.emit(str(exc))


class DetectorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOCraft - Detector")
        self.resize(1180, 720)

        self.model = None
        self.model_names = {}
        self.original_bgr = None
        self.annotated_bgr = None
        self.model_loader = None
        self.worker = None

        self.image_view = ImageView()
        self.image_view.file_dropped.connect(self.load_image)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self.image_view, stretch=3)
        layout.addWidget(self._build_panel(), stretch=2)
        self.setCentralWidget(central)

        self.statusBar().showMessage("Selecione um modelo para começar.")
        self._populate_models()
        self._update_controls()

    def _build_panel(self):
        panel = QWidget()
        panel.setMaximumWidth(420)
        layout = QVBoxLayout(panel)

        model_box = QGroupBox("Modelo")
        model_layout = QVBoxLayout(model_box)
        self.model_combo = QComboBox()
        self.model_combo.activated.connect(self._on_model_selected)
        self.browse_model_btn = QPushButton("Abrir modelo...")
        self.browse_model_btn.clicked.connect(self.browse_model)
        self.model_info = QLabel("Nenhum modelo carregado.")
        self.model_info.setWordWrap(True)
        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("Dispositivo:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("cpu")
        if torch.cuda.is_available():
            self.device_combo.addItem("cuda")
            self.device_combo.setToolTip(
                "A GPU pode estar ocupada com treinamento. CPU é mais seguro."
            )
        device_row.addWidget(self.device_combo, stretch=1)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.browse_model_btn)
        model_layout.addWidget(self.model_info)
        model_layout.addLayout(device_row)

        image_box = QGroupBox("Imagem")
        image_layout = QHBoxLayout(image_box)
        self.open_image_btn = QPushButton("Abrir imagem")
        self.open_image_btn.clicked.connect(self.browse_image)
        self.save_btn = QPushButton("Salvar resultado")
        self.save_btn.clicked.connect(self.save_result)
        image_layout.addWidget(self.open_image_btn)
        image_layout.addWidget(self.save_btn)

        detect_box = QGroupBox("Detecção")
        detect_layout = QVBoxLayout(detect_box)
        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Confiança:"))
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(25)
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        self.conf_slider.sliderReleased.connect(self._on_conf_released)
        self.conf_label = QLabel("0.25")
        self.conf_label.setMinimumWidth(36)
        conf_row.addWidget(self.conf_slider, stretch=1)
        conf_row.addWidget(self.conf_label)
        self.detect_btn = QPushButton("Detectar")
        self.detect_btn.clicked.connect(self.run_inference)
        detect_layout.addLayout(conf_row)
        detect_layout.addWidget(self.detect_btn)

        results_box = QGroupBox("Resultados")
        results_layout = QVBoxLayout(results_box)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Classe", "Confiança"])
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        results_layout.addWidget(self.table)
        results_layout.addWidget(self.summary_label)

        layout.addWidget(model_box)
        layout.addWidget(image_box)
        layout.addWidget(detect_box)
        layout.addWidget(results_box, stretch=1)
        return panel

    def _populate_models(self):
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItem("Selecione um modelo...", None)
        for path in discover_models(ROOT_DIR):
            self.model_combo.addItem(str(path.relative_to(ROOT_DIR)), str(path))
        self.model_combo.blockSignals(False)

    def _on_model_selected(self, index):
        path = self.model_combo.itemData(index)
        if path:
            self.load_model(path)

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar modelo", str(ROOT_DIR), "Pesos YOLO (*.pt)"
        )
        if path:
            existing = self.model_combo.findData(path)
            self.model_combo.blockSignals(True)
            if existing == -1:
                self.model_combo.addItem(str(Path(path).name), path)
                existing = self.model_combo.count() - 1
            self.model_combo.setCurrentIndex(existing)
            self.model_combo.blockSignals(False)
            self.load_model(path)

    def load_model(self, path):
        self.statusBar().showMessage(f"Carregando modelo: {Path(path).name} ...")
        self._set_busy(True)
        self.model_loader = ModelLoader(path)
        self.model_loader.loaded.connect(self._on_model_loaded)
        self.model_loader.failed.connect(self._on_model_failed)
        self.model_loader.start()

    def _on_model_loaded(self, model, names):
        self.model = model
        self.model_names = names
        self.model_info.setText(f"{model.ckpt_path or 'modelo'}\nClasses: {len(names)}")
        self.statusBar().showMessage(f"Modelo carregado. {len(names)} classes.")
        self._set_busy(False)

    def _on_model_failed(self, error):
        self._set_busy(False)
        QMessageBox.critical(self, "Erro ao carregar modelo", error)
        self.statusBar().showMessage("Falha ao carregar modelo.")

    def browse_image(self):
        start_dir = str(ROOT_DIR)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar imagem",
            start_dir,
            "Imagens (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)",
        )
        if path:
            self.load_image(path)

    def load_image(self, path):
        image = imread_unicode(path)
        if image is None:
            QMessageBox.warning(self, "Imagem inválida", f"Não foi possível abrir:\n{path}")
            return
        self.original_bgr = image
        self.annotated_bgr = None
        self.image_view.set_image(cv_to_qpixmap(image))
        self._clear_results()
        self.statusBar().showMessage(f"Imagem carregada: {Path(path).name}")
        self._update_controls()

    def _current_conf(self):
        return self.conf_slider.value() / 100.0

    def _on_conf_changed(self, value):
        self.conf_label.setText(f"{value / 100.0:.2f}")

    def _on_conf_released(self):
        if self.model is not None and self.original_bgr is not None and not self._is_busy():
            self.run_inference()

    def run_inference(self):
        if self.model is None:
            QMessageBox.information(self, "Sem modelo", "Carregue um modelo primeiro.")
            return
        if self.original_bgr is None:
            QMessageBox.information(self, "Sem imagem", "Carregue uma imagem primeiro.")
            return

        self._set_busy(True)
        self.statusBar().showMessage("Detectando...")
        self.worker = InferenceWorker(
            self.model,
            self.original_bgr,
            self._current_conf(),
            self.device_combo.currentText(),
        )
        self.worker.done.connect(self._on_inference_done)
        self.worker.failed.connect(self._on_inference_failed)
        self.worker.start()

    def _on_inference_done(self, annotated, detections, elapsed):
        self.annotated_bgr = annotated
        self.image_view.set_image(cv_to_qpixmap(annotated))
        self._fill_results(detections)
        self.statusBar().showMessage(
            f"{len(detections)} detecção(ões) em {elapsed * 1000:.0f} ms "
            f"({self.device_combo.currentText()})."
        )
        self._set_busy(False)

    def _on_inference_failed(self, error):
        self._set_busy(False)
        QMessageBox.critical(self, "Erro na detecção", error)
        self.statusBar().showMessage("Falha na detecção.")

    def _fill_results(self, detections):
        self.table.setRowCount(len(detections))
        counts = {}
        for row, (name, conf) in enumerate(detections):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            conf_item = QTableWidgetItem(f"{conf:.1%}")
            conf_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, 1, conf_item)
            counts[name] = counts.get(name, 0) + 1

        if counts:
            summary = ", ".join(f"{name}: {n}" for name, n in sorted(counts.items()))
            self.summary_label.setText(summary)
        else:
            self.summary_label.setText("Nenhuma detecção.")

    def _clear_results(self):
        self.table.setRowCount(0)
        self.summary_label.setText("")

    def save_result(self):
        if self.annotated_bgr is None:
            QMessageBox.information(self, "Nada para salvar", "Execute uma detecção primeiro.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Salvar resultado", str(ROOT_DIR / "deteccao.jpg"), "Imagem (*.jpg *.png)"
        )
        if path:
            if imwrite_unicode(path, self.annotated_bgr):
                self.statusBar().showMessage(f"Salvo em: {path}")
            else:
                QMessageBox.warning(self, "Erro", "Não foi possível salvar a imagem.")

    def _is_busy(self):
        return not self.detect_btn.isEnabled()

    def _set_busy(self, busy):
        self.detect_btn.setEnabled(not busy)
        self.browse_model_btn.setEnabled(not busy)
        self.open_image_btn.setEnabled(not busy)
        self.model_combo.setEnabled(not busy)
        if not busy:
            self._update_controls()

    def _update_controls(self):
        ready = self.model is not None and self.original_bgr is not None
        self.detect_btn.setEnabled(ready)
        self.save_btn.setEnabled(self.annotated_bgr is not None)

    def closeEvent(self, event):
        for thread in (self.model_loader, self.worker):
            if thread is not None and thread.isRunning():
                thread.wait(3000)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = DetectorWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
