"""Qt widget for real-time normal-map relighting of PCBA boards."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from .normal_relight import (
    RelightCase,
    RelightInputs,
    discover_relight_cases,
    load_relight_inputs,
    render_relight,
)


def _import_qt():
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except ImportError as error:
        raise ImportError("PySide6 is required for the Pseudo-3D Viewer tab.") from error
    return QtCore, QtGui, QtWidgets


QtCore, QtGui, QtWidgets = _import_qt()


class RelightImageLabel(QtWidgets.QLabel):
    """Image label that rescales the current pixmap on resize."""

    def __init__(self) -> None:
        super().__init__()
        self._source_pixmap: QtGui.QPixmap | None = None
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 420)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

    def set_source_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        self._source_pixmap = pixmap
        self._apply_scaled_pixmap()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802 - Qt API name
        super().resizeEvent(event)
        self._apply_scaled_pixmap()

    def _apply_scaled_pixmap(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        scaled = self._source_pixmap.scaled(
            self.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class Pseudo3DViewerWidget(QtWidgets.QWidget):
    """Reusable tab widget for relighting normal/albedo outputs."""

    def __init__(self, results_root: Path | str | None = None, max_render_edge: int = 1400) -> None:
        super().__init__()
        self.max_render_edge = max_render_edge
        self._cases: list[RelightCase] = []
        self._inputs: RelightInputs | None = None
        self._build_ui()
        if results_root is not None:
            self.root_edit.setText(str(Path(results_root)))
            self.scan_results()

    def _build_ui(self) -> None:
        self.setWindowTitle("Pseudo-3D Viewer")
        root_label = QtWidgets.QLabel("Results root")
        self.root_edit = QtWidgets.QLineEdit()
        self.root_edit.setPlaceholderText("Folder containing PCB result folders")
        browse_button = QtWidgets.QPushButton("Browse")
        browse_button.clicked.connect(self.browse_results_root)
        scan_button = QtWidgets.QPushButton("Scan")
        scan_button.clicked.connect(self.scan_results)

        root_row = QtWidgets.QHBoxLayout()
        root_row.addWidget(root_label)
        root_row.addWidget(self.root_edit, 1)
        root_row.addWidget(browse_button)
        root_row.addWidget(scan_button)

        self.board_list = QtWidgets.QListWidget()
        self.board_list.currentRowChanged.connect(self.load_selected_board)
        self.board_list.setMinimumWidth(180)

        self.azimuth_slider = self._slider(0, 360, 315)
        self.elevation_slider = self._slider(0, 90, 45)
        self.azimuth_value = QtWidgets.QLabel()
        self.elevation_value = QtWidgets.QLabel()
        self.azimuth_slider.valueChanged.connect(self.update_render)
        self.elevation_slider.valueChanged.connect(self.update_render)

        controls = QtWidgets.QFormLayout()
        controls.addRow("Azimuth", self._slider_row(self.azimuth_slider, self.azimuth_value))
        controls.addRow("Elevation", self._slider_row(self.elevation_slider, self.elevation_value))

        self.status_label = QtWidgets.QLabel("No PCBA result folder loaded.")
        self.status_label.setWordWrap(True)
        self.image_label = RelightImageLabel()

        side = QtWidgets.QVBoxLayout()
        side.addWidget(QtWidgets.QLabel("PCBA folders"))
        side.addWidget(self.board_list, 1)
        side.addLayout(controls)
        side.addWidget(self.status_label)

        body = QtWidgets.QHBoxLayout()
        body.addLayout(side)
        body.addWidget(self.image_label, 1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(root_row)
        layout.addLayout(body, 1)

    def _slider(self, minimum: int, maximum: int, value: int) -> QtWidgets.QSlider:
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.setSingleStep(1)
        slider.setPageStep(10)
        return slider

    def _slider_row(self, slider: QtWidgets.QSlider, value_label: QtWidgets.QLabel) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        value_label.setMinimumWidth(44)
        layout.addWidget(slider, 1)
        layout.addWidget(value_label)
        return row

    def browse_results_root(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select PCBA result folder", self.root_edit.text())
        if folder:
            self.root_edit.setText(folder)
            self.scan_results()

    def scan_results(self) -> None:
        try:
            self._cases = discover_relight_cases(Path(self.root_edit.text()))
        except (OSError, ValueError) as error:
            self._cases = []
            self.board_list.clear()
            self.status_label.setText(str(error))
            return
        self.board_list.clear()
        for case in self._cases:
            print(f"PCBA folder: {case.name} | normal={case.normal_path.name} | albedo={case.albedo_path.name}")
            item = QtWidgets.QListWidgetItem(case.name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, str(case.folder))
            self.board_list.addItem(item)
        self.status_label.setText(f"Found {len(self._cases)} PCBA folders.")
        if self._cases:
            self.board_list.setCurrentRow(0)

    def load_selected_board(self, row: int) -> None:
        if row < 0 or row >= len(self._cases):
            return
        case = self._cases[row]
        try:
            self._inputs = load_relight_inputs(case, max_edge=self.max_render_edge)
        except (OSError, ValueError) as error:
            self._inputs = None
            self.status_label.setText(f"{case.name}: {error}")
            return
        normal_shape = self._inputs.normals.shape[:2]
        self.status_label.setText(
            f"{case.name}: {normal_shape[1]}x{normal_shape[0]} relight view from "
            f"{case.normal_path.name} + {case.albedo_path.name}"
        )
        self.update_render()

    def update_render(self) -> None:
        self.azimuth_value.setText(f"{self.azimuth_slider.value()} deg")
        self.elevation_value.setText(f"{self.elevation_slider.value()} deg")
        if self._inputs is None:
            return
        rgb = render_relight(
            self._inputs.normals,
            self._inputs.albedo,
            self.azimuth_slider.value(),
            self.elevation_slider.value(),
        )
        self.image_label.set_source_pixmap(QtGui.QPixmap.fromImage(qimage_from_rgb(rgb)))


def qimage_from_rgb(rgb: np.ndarray) -> QtGui.QImage:
    array = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width, channels = array.shape
    if channels != 3:
        raise ValueError(f"RGB image must have 3 channels, got {channels}")
    image = QtGui.QImage(array.data, width, height, width * channels, QtGui.QImage.Format.Format_RGB888)
    return image.copy()


def create_pseudo3d_tab(results_root: Path | str | None = None) -> Pseudo3DViewerWidget:
    """Factory used by an existing dashboard QTabWidget."""

    return Pseudo3DViewerWidget(results_root=results_root)


def add_pseudo3d_tab(
    tabs: QtWidgets.QTabWidget,
    results_root: Path | str | None = None,
    title: str = "Pseudo-3D Viewer",
) -> Pseudo3DViewerWidget:
    """Create and attach the viewer to an existing dashboard tab widget."""

    widget = create_pseudo3d_tab(results_root)
    tabs.addTab(widget, title)
    return widget


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Open the FabLoop Pseudo-3D Viewer.")
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--max-render-edge", type=int, default=1400)
    args = parser.parse_args(argv)

    app = QtWidgets.QApplication(sys.argv[:1])
    widget = Pseudo3DViewerWidget(args.results_root, max_render_edge=args.max_render_edge)
    widget.resize(1280, 820)
    widget.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
