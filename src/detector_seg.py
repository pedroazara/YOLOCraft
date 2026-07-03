import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtNetwork import QLocalServer
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
from ultralytics import SAM, YOLO

from sam_segmentation import SamMobSegmenter
from segmentation import MobSegmenter

ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
DETECTOR_SERVER = "yolocraft_detector"


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
            if path not in seen and "sam" not in path.name.lower():
                seen.add(path)
                found.append(path)
    return found


def discover_sam_weights(root):
    patterns = [
        "pretrained_models/*sam*.pt",
        "models/*sam*.pt",
        "**/*sam*.pt",
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
    """Carrega o modelo de detecção (YOLO)."""

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


class SamLoader(QThread):
    """Carrega os pesos do SAM/MobileSAM (mais pesado, por isso feito sob demanda)."""

    loaded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            sam = SAM(self.path)
            self.loaded.emit(sam)
        except Exception as exc:
            self.failed.emit(str(exc))


class InferenceWorker(QThread):
    """Roda a detecção e, dependendo do modo, também a segmentação (clássica ou SAM)."""

    done = pyqtSignal(object, list, float)
    failed = pyqtSignal(str)

    def __init__(
        self,
        model,
        image_bgr,
        conf,
        device,
        mode,
        classic_method,
        classic_segmenter,
        sam_segmenter,
    ):
        super().__init__()
        self.model = model
        self.image_bgr = image_bgr
        self.conf = conf
        self.device = device
        self.mode = mode                        # "none" | "classic" | "sam"
        self.classic_method = classic_method     # "auto" | "otsu" | "hsv" | "grabcut"
        self.classic_segmenter = classic_segmenter
        self.sam_segmenter = sam_segmenter

    def run(self):
        try:
            start = time.time()

            if self.mode == "classic" and self.classic_segmenter is not None:
                self.classic_segmenter.conf_threshold = self.conf
                result = self.classic_segmenter.detect_and_segment(
                    self.image_bgr, method=self.classic_method
                )
                annotated = self.classic_segmenter.draw_detections(
                    self.image_bgr, result["detections"]
                )
                detections = [(d["class"], d["confidence"]) for d in result["detections"]]

            elif self.mode == "sam" and self.sam_segmenter is not None:
                self.sam_segmenter.conf_threshold = self.conf
                result = self.sam_segmenter.detect_and_segment(self.image_bgr)
                annotated = self.sam_segmenter.draw_detections(
                    self.image_bgr, result["detections"]
                )
                detections = [(d["class"], d["confidence"]) for d in result["detections"]]

            else:
                result = self.model.predict(
                    source=self.image_bgr,
                    conf=self.conf,
                    device=self.device,
                    verbose=False,
                )[0]
                annotated = np.ascontiguousarray(result.plot())
                detections = [
                    (result.names[int(box.cls[0])], float(box.conf[0]))
                    for box in result.boxes
                ]

            detections.sort(key=lambda d: d[1], reverse=True)
            elapsed = time.time() - start
            self.done.emit(annotated, detections, elapsed)
        except Exception as exc:
            self.failed.emit(str(exc))


class DetectorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOCraft - Detector + Segmentação")
        self.resize(1220, 760)

        self.model = None
        self.model_names = {}
        self.original_bgr = None
        self.annotated_bgr = None
        self.model_loader = None
        self.worker = None

        self.classic_segmenter = None
        self.sam_segmenter = None
        self._raw_sam_model = None
        self.sam_loader = None

        self.image_view = ImageView()
        self.image_view.file_dropped.connect(self.load_image)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self.image_view, stretch=3)
        layout.addWidget(self._build_panel(), stretch=2)
        self.setCentralWidget(central)

        self.statusBar().showMessage("Selecione um modelo para começar.")
        self._populate_models()
        self._populate_sam_weights()
        self._update_controls()
        self._setup_server()

    def _setup_server(self):
        self.server = QLocalServer(self)
        QLocalServer.removeServer(DETECTOR_SERVER)
        if self.server.listen(DETECTOR_SERVER):
            self.server.newConnection.connect(self._on_new_connection)
        else:
            self.server = None

    def _on_new_connection(self):
        socket = self.server.nextPendingConnection()
        if socket is None:
            return
        if socket.waitForReadyRead(1000):
            path = bytes(socket.readAll()).decode("utf-8", errors="replace").strip()
            if path:
                self.load_image(path)
                self.showNormal()
                self.raise_()
                self.activateWindow()
        socket.disconnectFromServer()

    def _build_panel(self):
        panel = QWidget()
        panel.setMaximumWidth(440)
        layout = QVBoxLayout(panel)

        model_box = QGroupBox("Modelo de detecção (YOLO)")
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
            "Válido apenas no modo 'Sem segmentação'. Nos modos com segmentação "
            "(Clássica/SAM), o dispositivo usado é o padrão do modelo carregado."
        )
        device_row.addWidget(self.device_combo, stretch=1)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.browse_model_btn)
        model_layout.addWidget(self.model_info)
        model_layout.addLayout(device_row)

        seg_box = QGroupBox("Segmentação")
        seg_layout = QVBoxLayout(seg_box)

        self.segmentation_combo = QComboBox()
        self.segmentation_combo.addItem("Sem segmentação (apenas boxes)", "none")
        self.segmentation_combo.addItem("Clássica (Otsu / HSV / GrabCut)", "classic")
        self.segmentation_combo.addItem("SAM (MobileSAM)", "sam")
        self.segmentation_combo.activated.connect(self._on_segmentation_mode_changed)
        seg_layout.addWidget(self.segmentation_combo)

        self.classic_method_label = QLabel("Método clássico:")
        self.classic_method_combo = QComboBox()
        self.classic_method_combo.addItem("Auto", "auto")
        self.classic_method_combo.addItem("GrabCut", "grabcut")
        self.classic_method_combo.addItem("HSV", "hsv")
        self.classic_method_combo.addItem("Otsu", "otsu")
        self.classic_method_label.setVisible(False)
        self.classic_method_combo.setVisible(False)
        seg_layout.addWidget(self.classic_method_label)
        seg_layout.addWidget(self.classic_method_combo)

        self.sam_row_widget = QWidget()
        sam_row_layout = QHBoxLayout(self.sam_row_widget)
        sam_row_layout.setContentsMargins(0, 0, 0, 0)
        self.sam_combo = QComboBox()
        self.sam_combo.activated.connect(self._on_sam_selected)
        self.sam_browse_btn = QPushButton("Abrir SAM...")
        self.sam_browse_btn.clicked.connect(self.browse_sam)
        sam_row_layout.addWidget(self.sam_combo, stretch=1)
        sam_row_layout.addWidget(self.sam_browse_btn)
        self.sam_row_widget.setVisible(False)
        seg_layout.addWidget(self.sam_row_widget)

        self.sam_status_label = QLabel("")
        self.sam_status_label.setWordWrap(True)
        self.sam_status_label.setVisible(False)
        seg_layout.addWidget(self.sam_status_label)

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
        layout.addWidget(seg_box)
        layout.addWidget(image_box)
        layout.addWidget(detect_box)
        layout.addWidget(results_box, stretch=1)
        return panel

    # ------------------------------------------------------------------
    # Modelo de detecção (YOLO)
    # ------------------------------------------------------------------
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
        self._rebuild_segmenters()
        self.statusBar().showMessage(f"Modelo carregado. {len(names)} classes.")
        self._set_busy(False)

    def _on_model_failed(self, error):
        self._set_busy(False)
        QMessageBox.critical(self, "Erro ao carregar modelo", error)
        self.statusBar().showMessage("Falha ao carregar modelo.")

    # ------------------------------------------------------------------
    # Segmentação (clássica / SAM)
    # ------------------------------------------------------------------
    def _rebuild_segmenters(self):
        """Reconstrói os wrappers de segmentação reaproveitando o YOLO já carregado."""
        if self.model is None:
            self.classic_segmenter = None
            self.sam_segmenter = None
            return
        self.classic_segmenter = MobSegmenter(self.model, default_method="auto")
        if self._raw_sam_model is not None:
            self.sam_segmenter = SamMobSegmenter(self.model, self._raw_sam_model)
        else:
            self.sam_segmenter = None

    def _on_segmentation_mode_changed(self, index):
        mode = self.segmentation_combo.itemData(index)
        self.classic_method_label.setVisible(mode == "classic")
        self.classic_method_combo.setVisible(mode == "classic")
        self.sam_row_widget.setVisible(mode == "sam")
        self.sam_status_label.setVisible(mode == "sam")

        if mode == "sam" and self.sam_segmenter is None:
            sam_path = self.sam_combo.currentData()
            if sam_path:
                self._load_sam(sam_path)
            else:
                self.sam_status_label.setText(
                    "Selecione (ou abra) os pesos do SAM acima."
                )
        self._update_controls()

    def _populate_sam_weights(self):
        self.sam_combo.blockSignals(True)
        self.sam_combo.clear()
        self.sam_combo.addItem("Selecione os pesos do SAM...", None)
        for path in discover_sam_weights(ROOT_DIR):
            self.sam_combo.addItem(str(path.relative_to(ROOT_DIR)), str(path))
        self.sam_combo.blockSignals(False)

    def _on_sam_selected(self, index):
        path = self.sam_combo.itemData(index)
        if path:
            self._raw_sam_model = None
            self.sam_segmenter = None
            self._load_sam(path)

    def browse_sam(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Selecionar pesos do SAM", str(ROOT_DIR), "Pesos SAM (*.pt)"
        )
        if path:
            existing = self.sam_combo.findData(path)
            self.sam_combo.blockSignals(True)
            if existing == -1:
                self.sam_combo.addItem(str(Path(path).name), path)
                existing = self.sam_combo.count() - 1
            self.sam_combo.setCurrentIndex(existing)
            self.sam_combo.blockSignals(False)
            self._raw_sam_model = None
            self.sam_segmenter = None
            self._load_sam(path)

    def _load_sam(self, path):
        self.sam_status_label.setVisible(True)
        self.sam_status_label.setText(f"Carregando SAM: {Path(path).name} ...")
        self.statusBar().showMessage(f"Carregando SAM: {Path(path).name} ...")
        self._set_busy(True)
        self.sam_loader = SamLoader(path)
        self.sam_loader.loaded.connect(self._on_sam_loaded)
        self.sam_loader.failed.connect(self._on_sam_failed)
        self.sam_loader.start()

    def _on_sam_loaded(self, sam_model):
        self._raw_sam_model = sam_model
        self._rebuild_segmenters()
        self.sam_status_label.setText("SAM carregado ✔")
        self.statusBar().showMessage("SAM carregado.")
        self._set_busy(False)

    def _on_sam_failed(self, error):
        self._set_busy(False)
        self.sam_status_label.setText("Falha ao carregar SAM.")
        QMessageBox.critical(self, "Erro ao carregar SAM", error)
        self.statusBar().showMessage("Falha ao carregar SAM.")
        # evita deixar a interface travada no modo SAM sem modelo carregado
        self.segmentation_combo.setCurrentIndex(0)
        self._on_segmentation_mode_changed(0)

    # ------------------------------------------------------------------
    # Imagem
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Inferência
    # ------------------------------------------------------------------
    def run_inference(self):
        if self.model is None:
            QMessageBox.information(self, "Sem modelo", "Carregue um modelo primeiro.")
            return
        if self.original_bgr is None:
            QMessageBox.information(self, "Sem imagem", "Carregue uma imagem primeiro.")
            return

        mode = self.segmentation_combo.currentData()
        if mode == "sam" and self.sam_segmenter is None:
            QMessageBox.information(
                self, "SAM não carregado",
                "Aguarde o SAM terminar de carregar (ou selecione outro modo de segmentação).",
            )
            return

        self._set_busy(True)
        self.statusBar().showMessage("Detectando...")
        self.worker = InferenceWorker(
            self.model,
            self.original_bgr,
            self._current_conf(),
            self.device_combo.currentText(),
            mode,
            self.classic_method_combo.currentData(),
            self.classic_segmenter,
            self.sam_segmenter,
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

    # ------------------------------------------------------------------
    # Estado da interface
    # ------------------------------------------------------------------
    def _is_busy(self):
        return not self.detect_btn.isEnabled()

    def _set_busy(self, busy):
        self.detect_btn.setEnabled(not busy)
        self.browse_model_btn.setEnabled(not busy)
        self.open_image_btn.setEnabled(not busy)
        self.model_combo.setEnabled(not busy)
        self.segmentation_combo.setEnabled(not busy)
        self.sam_combo.setEnabled(not busy)
        self.sam_browse_btn.setEnabled(not busy)
        if not busy:
            self._update_controls()

    def _update_controls(self):
        mode = self.segmentation_combo.currentData() if hasattr(self, "segmentation_combo") else "none"
        sam_ready = mode != "sam" or self.sam_segmenter is not None
        ready = self.model is not None and self.original_bgr is not None and sam_ready
        self.detect_btn.setEnabled(ready)
        self.save_btn.setEnabled(self.annotated_bgr is not None)

    def closeEvent(self, event):
        if getattr(self, "server", None) is not None:
            self.server.close()
        for thread in (self.model_loader, self.sam_loader, self.worker):
            if thread is not None and thread.isRunning():
                thread.wait(3000)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = DetectorWindow()
    window.show()
    for arg in sys.argv[1:]:
        if Path(arg).is_file():
            window.load_image(arg)
            break
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
