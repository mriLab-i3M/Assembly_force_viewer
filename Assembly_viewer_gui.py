# -*- coding: utf-8 -*-
# Copilot: extend this GUI with a system selector and path manager
import sys
import os
import re
import json
import subprocess
import tempfile
import numpy as np
import pandas as pd

from custom_lids import (
    CUSTOM_TABLE_NAME,
    CUSTOM_VISUALIZATION_NAME,
    build_custom_lid_config_path,
    load_custom_lid_config,
    make_preview_png,
    save_custom_lid_config,
)
from table_utils import (
    LID_TABLE_COLUMNS,
    LID_VALUE_COLUMNS,
    build_lid_pivot,
    component_column,
    extreme_cells,
    filter_lid_rows,
    find_worst_lid_case,
    is_norm_column,
    lid_sort_key,
    read_lid_table,
    sorted_lids,
    sorted_rings,
)
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QPushButton, QLabel,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QFileDialog, QMessageBox, QInputDialog, QDialog,
    QCheckBox, QFormLayout, QLineEdit, QDialogButtonBox,
    QGroupBox, QScrollArea, QFrame, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import Qt, pyqtSignal


# =========================================================
# CONFIG
# =========================================================

STUDY_OFF = "OFF"
LEGACY_STUDY = "Legacy Lids (+Y/-Y/+Z/-Z)"
VIEW_ALL = "All data"
VIEW_COMPONENTS = ["Fx", "Fy", "Fz", "NormF", "Tx", "Ty", "Tz", "NormT"]
VIEW_LID_COMPONENTS = [f"{name}_Lids" for name in VIEW_COMPONENTS]
LEGACY_PERLID_METRICS = ["PerLid", "PerLid+Y", "PerLid-Y", "PerLid+Z", "PerLid-Z"]
BASE_METRICS = ["SumPerRing", "PerCube", "Mounting_Ring"]

COLOR_MAX = QColor(255, 100, 100)
COLOR_MIN = QColor(110, 160, 255)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(_SCRIPT_DIR, "systems_config.json")

DEFAULT_SYSTEMS = {
    "Next": r"Z:\Projects\PhysioII - NextMRI\Magnet\Forces\Estudio fuerzas montaje",
    "Preclinico": r"Z:\Projects\Preclinico\Estudio fuerzas montaje",
}


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return dict(DEFAULT_SYSTEMS)


def save_config(systems):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(systems, f, indent=2, ensure_ascii=False)


# =========================================================
# BARRA DE FILTROS (CHECKBOXES)
# =========================================================

class CheckFilterBar(QWidget):
    """Row of checkboxes (one per item) with All/None buttons; all items start checked."""

    changed = pyqtSignal()

    def __init__(self, title):
        super().__init__()
        self._boxes = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        title_label = QLabel(title)
        title_label.setMinimumWidth(45)
        self.all_button = QPushButton("All")
        self.none_button = QPushButton("None")
        for button in (self.all_button, self.none_button):
            button.setFixedSize(44, 20)
            button.setStyleSheet("padding: 0px;")

        self._holder = QWidget()
        self._holder_layout = QHBoxLayout(self._holder)
        self._holder_layout.setContentsMargins(4, 0, 4, 0)
        self._holder_layout.setSpacing(8)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._holder)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QCheckBox { padding: 0px; margin: 0px; } QScrollBar:horizontal { height: 8px; }")
        self._scroll.setFixedHeight(self._row_height())

        layout.addWidget(title_label)
        layout.addWidget(self.all_button)
        layout.addWidget(self.none_button)
        layout.addWidget(self._scroll, 1)

        self.all_button.clicked.connect(lambda: self._set_all(True))
        self.none_button.clicked.connect(lambda: self._set_all(False))
        self.setEnabled(False)

    def _row_height(self):
        """Just enough for one row of checkboxes plus the horizontal scrollbar (when it is needed)."""
        box_height = QCheckBox("Ring 00").sizeHint().height()
        return box_height + 8 + 2

    def items(self):
        return [box.text() for box in self._boxes]

    def selected(self):
        return [box.text() for box in self._boxes if box.isChecked()]

    def set_items(self, items):
        """Replace the options (all checked). Keeps current checks if the options are unchanged."""
        items = [str(item) for item in items]
        if items == self.items():
            return

        while self._holder_layout.count():
            child = self._holder_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self._boxes = []

        for item in items:
            box = QCheckBox(item)
            box.setChecked(True)
            box.toggled.connect(self.changed)
            self._holder_layout.addWidget(box)
            self._boxes.append(box)
        self._holder_layout.addStretch()
        self.setEnabled(bool(items))

    def _set_all(self, checked):
        for box in self._boxes:
            box.blockSignals(True)
            box.setChecked(checked)
            box.blockSignals(False)
        self.changed.emit()


# =========================================================
# VISOR CON ZOOM PERSISTENTE
# =========================================================

class ImageViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.first_load = True

    def wheelEvent(self, event):
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 0.85
        self.scale(zoom_factor, zoom_factor)


# =========================================================
# MAIN VIEWER
# =========================================================

class StepViewer(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Visualizador Fuerzas Montaje")
        self.resize(1700, 1000)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.base_folder = None
        self.lids_available = False

        # =======================
        # SYSTEM SELECTOR
        # =======================

        self._systems = load_config()
        self._updating_system = False

        system_layout = QHBoxLayout()

        self.system_selector = QComboBox()
        self.system_selector.addItem("-- Select System --")
        for name in self._systems:
            self.system_selector.addItem(name)

        self.add_system_button = QPushButton("Add System")
        self.run_analysis_button = QPushButton("Run Force Assembly Analysis")
        self.custom_lids_button = QPushButton("Add Custom Lids")

        system_layout.addWidget(QLabel("System:"))
        system_layout.addWidget(self.system_selector)
        system_layout.addWidget(self.add_system_button)
        system_layout.addWidget(self.run_analysis_button)
        system_layout.addWidget(self.custom_lids_button)
        system_layout.addStretch()

        self.layout.addLayout(system_layout)

        # =======================
        # SELECTOR CARPETA
        # =======================

        folder_layout = QHBoxLayout()

        self.folder_display = QLabel("No seleccionada")
        self.browse_button = QPushButton("Browse")

        folder_layout.addWidget(QLabel("Carpeta base:"))
        folder_layout.addWidget(self.folder_display)
        folder_layout.addWidget(self.browse_button)

        self.layout.addLayout(folder_layout)
        self.browse_button.clicked.connect(self.select_base_folder)

        # =======================
        # STEP + SELECTORES
        # =======================

        top_layout = QHBoxLayout()

        self.prev_button = QPushButton("⬅")
        self.next_button = QPushButton("➡")
        self.step_selector = QComboBox()

        self.plot_selector = QComboBox()
        self.plot_selector.addItems([
            "Fx", "Fy", "Fz", "NormF",
            "Tx", "Ty", "Tz", "NormT"
        ])

        # -------- LIDS VISUALIZATION: estudio -> vista (en cascada) --------

        self.lids_label = QLabel("Lids visualization:")

        self.lids_study_selector = QComboBox()
        self.lids_study_selector.addItem(STUDY_OFF)
        self.lids_study_selector.setMinimumWidth(320)

        self.lids_view_selector = QComboBox()
        self.lids_view_selector.addItem(VIEW_ALL)
        self.lids_view_selector.setEnabled(False)
        self.lids_view_selector.setMinimumWidth(130)

        self._table_spec = None

        top_layout.addWidget(self.prev_button)
        top_layout.addWidget(self.step_selector)
        top_layout.addWidget(self.next_button)
        top_layout.addWidget(self.plot_selector)
        top_layout.addWidget(self.lids_label)
        top_layout.addWidget(self.lids_study_selector)
        top_layout.addWidget(self.lids_view_selector)

        self.layout.addLayout(top_layout)

        # =======================
        # WORST CASE
        # =======================

        worst_layout = QHBoxLayout()

        self.metric_type = QComboBox()
        self.metric_type.addItems(BASE_METRICS)

        self.metric_component = QComboBox()
        self.metric_component.addItems([
            "Fx", "Fy", "Fz", "normF",
            "Tx", "Ty", "Tz", "normT"
        ])

        self.worst_button = QPushButton("Find Worst Step")
        self.worst_result = QLabel("")

        worst_layout.addWidget(QLabel("Worst case:"))
        worst_layout.addWidget(self.metric_type)
        worst_layout.addWidget(self.metric_component)
        worst_layout.addWidget(self.worst_button)
        worst_layout.addWidget(self.worst_result)

        self.layout.addLayout(worst_layout)

        # =======================
        # RESET ZOOM
        # =======================

        self.reset_zoom_button = QPushButton("Reset Zoom")
        self.layout.addWidget(self.reset_zoom_button)
        
        self.reset_view_button = QPushButton("Reset View")
        self.layout.addWidget(self.reset_view_button)
        # =======================
        # IMAGEN
        # =======================

        self.viewer = ImageViewer()
        self.layout.addWidget(self.viewer)

        # =======================
        # TABLA
        # =======================

        # Contenedor compacto: solo crece en horizontal, altura estricta del contenido
        self.filter_box = QGroupBox("Table filters")
        filter_layout = QVBoxLayout(self.filter_box)
        filter_layout.setContentsMargins(6, 2, 6, 2)
        filter_layout.setSpacing(2)
        self.ring_filter = CheckFilterBar("Rings:")
        self.lid_filter = CheckFilterBar("Lids:")
        filter_layout.addWidget(self.ring_filter)
        filter_layout.addWidget(self.lid_filter)
        self.filter_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        filter_holder = QWidget()
        holder_layout = QVBoxLayout(filter_holder)
        holder_layout.setContentsMargins(0, 2, 0, 5)
        holder_layout.addWidget(self.filter_box)
        filter_holder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.layout.addWidget(filter_holder, 0)

        self.table = QTableWidget()
        self.layout.addWidget(self.table)

        # Conexiones
        self.prev_button.clicked.connect(self.go_previous)
        self.next_button.clicked.connect(self.go_next)
        self.plot_selector.currentIndexChanged.connect(self.on_plot_changed)
        self.lids_study_selector.currentIndexChanged.connect(self.on_lids_study_changed)
        self.lids_view_selector.currentIndexChanged.connect(self.update_view)
        self.ring_filter.changed.connect(self.render_table)
        self.lid_filter.changed.connect(self.render_table)
        self.worst_button.clicked.connect(self.compute_worst_case)
        self.reset_zoom_button.clicked.connect(self.reset_zoom)
        self.reset_view_button.clicked.connect(self.reset_view)
        self.system_selector.currentIndexChanged.connect(self.on_system_selected)
        self.add_system_button.clicked.connect(self.on_add_system)
        self.run_analysis_button.clicked.connect(self.on_run_analysis)
        self.custom_lids_button.clicked.connect(self.open_custom_lids_panel)
        self.step_selector.currentIndexChanged.connect(self.refresh_lids_study_selector)
        

    # =========================================================
    def reset_view(self):

        if not self.base_folder:
            return

        # Ir al primer Step
        if self.step_selector.count() > 0:
            self.step_selector.setCurrentIndex(0)

        # Primer plot
        self.plot_selector.setCurrentIndex(0)

        # Desactivar Lids
        self.lids_study_selector.blockSignals(True)
        self.lids_study_selector.setCurrentIndex(0)
        self.lids_study_selector.blockSignals(False)
        self._sync_lids_view_selector()
        self._refresh_lids_metric_types()

        # Limpiar worst case
        self.worst_result.setText("")

        # Reset zoom
        self.viewer.resetTransform()
        self.viewer.first_load = True

        # Forzar actualización
        self.update_view()

    def select_base_folder(self):

        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta", "")

        if folder:
            self.base_folder = folder
            self.folder_display.setText(folder)
            self.load_steps()

    # =========================================================

    def load_steps(self):

        self.step_selector.clear()

        if not self.base_folder:
            return

        steps = [
            f for f in os.listdir(self.base_folder)
            if os.path.isdir(os.path.join(self.base_folder, f))
            and f.startswith("Step")
        ]

        def extract_num(name):
            match = re.search(r"Step(\d+)", name)
            return int(match.group(1)) if match else -1

        steps_sorted = sorted(steps, key=extract_num)
        self.step_selector.addItems(steps_sorted)

        # ===== Detectar Lids =====
        self.lids_available = False

        for step in steps_sorted:
            test_path = os.path.join(
                self.base_folder,
                step,
                f"{step}_TableForceTorqueSum_perLid.txt"
            )
            if os.path.exists(test_path):
                self.lids_available = True
                break

        self.refresh_lids_study_selector()

        if steps_sorted:
            self.update_view()

    def _custom_study_folders(self, step_path):
        if not os.path.isdir(step_path):
            return []
        return [
            entry for entry in sorted(os.listdir(step_path))
            if os.path.isdir(os.path.join(step_path, entry)) and re.search(r"\d+Lids_", entry)
        ]

    def refresh_lids_study_selector(self):
        """First dropdown: OFF + custom studies of the active Step (legacy Lid+Y/Z only as fallback)."""
        previous = self.lids_study_selector.currentText()

        entries = []
        step = self.step_selector.currentText()
        if self.base_folder and step:
            step_path = os.path.join(self.base_folder, step)
            entries = self._custom_study_folders(step_path)
            legacy_table = os.path.join(step_path, f"{step}_TableForceTorqueSum_perLid.txt")
            if not entries and os.path.exists(legacy_table):
                entries = [LEGACY_STUDY]

        self.lids_study_selector.blockSignals(True)
        self.lids_study_selector.clear()
        self.lids_study_selector.addItem(STUDY_OFF)
        self.lids_study_selector.addItems(entries)
        for i, entry in enumerate(entries, start=1):
            self.lids_study_selector.setItemData(i, entry, Qt.ToolTipRole)
        if previous in entries:
            self.lids_study_selector.setCurrentText(previous)
        self.lids_study_selector.blockSignals(False)

        self._sync_lids_view_selector()
        self._refresh_lids_metric_types()
        self.update_view()

    def on_plot_changed(self):
        """The left (global) dropdown is always active: using it returns to the general view (Lids OFF)."""
        if self.lids_study_selector.currentIndex() > 0:
            self.lids_study_selector.setCurrentIndex(0)
            return
        self.update_view()

    def on_lids_study_changed(self):
        self._sync_lids_view_selector()
        self._refresh_lids_metric_types()
        self.update_view()

    def _study_state(self, step_path):
        """('off' | 'legacy' | 'custom', folder of the study)."""
        name = self.lids_study_selector.currentText()
        if name == LEGACY_STUDY:
            return "legacy", step_path
        if name not in ("", STUDY_OFF):
            path = os.path.join(step_path, name)
            if os.path.isdir(path):
                return "custom", path
        return "off", None

    def _sync_lids_view_selector(self):
        """Second dropdown: disabled while the study is OFF; "All data" by default once enabled."""
        study_on = self.lids_study_selector.currentText() not in ("", STUDY_OFF)
        previous = self.lids_view_selector.currentText()

        self.lids_view_selector.blockSignals(True)
        self.lids_view_selector.clear()
        self.lids_view_selector.addItem(VIEW_ALL)
        if study_on:
            self.lids_view_selector.addItems(VIEW_LID_COMPONENTS)
            if self.lids_view_selector.findText(previous) >= 0:
                self.lids_view_selector.setCurrentText(previous)
        self.lids_view_selector.setEnabled(study_on)
        self.lids_view_selector.blockSignals(False)


    def _refresh_lids_metric_types(self):
        """Worst-case PerLid options: Lid 1..N for a custom study, Lid+Y/... only for legacy studies."""
        previous = self.metric_type.currentText()
        extra = []

        step = self.step_selector.currentText()
        if self.base_folder and step:
            kind, path = self._study_state(os.path.join(self.base_folder, step))
            has_custom = any(
                self.lids_study_selector.itemText(i) not in (STUDY_OFF, LEGACY_STUDY)
                for i in range(self.lids_study_selector.count())
            )
            if kind == "custom":
                table = os.path.join(path, CUSTOM_TABLE_NAME)
                lids = sorted_lids(read_lid_table(table)["Lid"]) if os.path.exists(table) else []
                extra = ["PerLid"] + [f"PerLid {lid}" for lid in lids]
            elif not has_custom and self.lids_available:
                extra = LEGACY_PERLID_METRICS

        self.metric_type.clear()
        self.metric_type.addItems(BASE_METRICS + extra)
        if self.metric_type.findText(previous) >= 0:
            self.metric_type.setCurrentText(previous)
    def open_custom_lids_panel(self):
        if not self.base_folder:
            QMessageBox.warning(self, "No system", "Select a system first.")
            return

        dialog = CustomLidsDialog(self)
        dialog.exec_()

        self.refresh_lids_study_selector()

    # =========================================================

    def go_previous(self):
        i = self.step_selector.currentIndex()
        if i > 0:
            self.step_selector.setCurrentIndex(i - 1)

    def go_next(self):
        i = self.step_selector.currentIndex()
        if i < self.step_selector.count() - 1:
            self.step_selector.setCurrentIndex(i + 1)

    # =========================================================

    def reset_zoom(self):
        self.viewer.resetTransform()
        self.viewer.first_load = True
        self.update_view()

    # =========================================================

    def update_view(self):

        if not self.base_folder:
            return

        step = self.step_selector.currentText()
        plot = self.plot_selector.currentText()
        view = self.lids_view_selector.currentText()

        step_path = os.path.join(self.base_folder, step)
        kind, study_path = self._study_state(step_path)

        self._show_image(self._resolve_image_path(step, step_path, plot, kind, study_path, view))

        self._table_spec = self._build_table_spec(step, step_path, kind, study_path, view)
        self._sync_filters()
        self.render_table()

        plt.close("all")

    # =========================================================
    # IMAGEN
    # =========================================================

    def _resolve_image_path(self, step, step_path, plot, kind, study_path, view):
        if kind == "off":
            return os.path.join(step_path, f"{step}_{plot}.png")

        if view == VIEW_ALL:
            if kind == "custom":
                return os.path.join(study_path, CUSTOM_VISUALIZATION_NAME)
            return os.path.join(step_path, f"{step}_{plot}.png")

        return os.path.join(study_path, f"{step}_{view}.png")

    def _show_image(self, image_path):
        self.viewer.scene().clear()

        if os.path.exists(image_path):

            pixmap = QPixmap(image_path)
            item = QGraphicsPixmapItem(pixmap)
            self.viewer.scene().addItem(item)
            self.viewer.setSceneRect(item.boundingRect())

            if self.viewer.first_load:
                self.viewer.fitInView(item, Qt.KeepAspectRatio)
                self.viewer.first_load = False

    # =========================================================
    # TABLA: datos, filtros y render
    # =========================================================

    def _build_table_spec(self, step, step_path, kind, study_path, view):
        """Describe the table to show: {'kind': 'lids_all' | 'lids_pivot' | 'rings', 'df': ..., ...}."""

        if kind != "off":
            if kind == "custom":
                table_path = os.path.join(study_path, CUSTOM_TABLE_NAME)
            else:
                table_path = os.path.join(step_path, f"{step}_TableForceTorqueSum_perLid.txt")

            if not os.path.exists(table_path):
                return None

            df = read_lid_table(table_path)
            if view != VIEW_ALL and df["Ring"].notna().any():
                return {"kind": "lids_pivot", "df": df, "component": view}
            return {"kind": "lids_all", "df": df}

        table_path = os.path.join(step_path, f"{step}_TableForceTorqueSum_perRing.txt")
        if not os.path.exists(table_path):
            return None

        df = pd.read_csv(table_path, sep="\t", decimal=",")
        df["Ring"] = pd.to_numeric(df["Ring"], errors="coerce").astype("Int64")

        return {"kind": "rings", "df": df, "summary": True}

    def _sync_filters(self):
        """Fill Rings/Lids filters from the loaded table (everything checked by default)."""
        spec = self._table_spec
        rings, lids = [], []
        if spec is not None:
            df = spec["df"]
            rings = sorted_rings(df["Ring"].dropna())
            if "Lid" in df.columns:
                lids = sorted_lids(df["Lid"])

        self.ring_filter.blockSignals(True)
        self.lid_filter.blockSignals(True)
        self.ring_filter.set_items(rings)
        self.lid_filter.set_items(lids)
        self.ring_filter.blockSignals(False)
        self.lid_filter.blockSignals(False)

    def render_table(self):
        """Apply the Rings/Lids selections to the current table and draw it with max/min highlighting."""
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)

        spec = self._table_spec
        if spec is None:
            return

        rings = [int(r) for r in self.ring_filter.selected()] if self.ring_filter.items() else None
        lids = self.lid_filter.selected() if self.lid_filter.items() else None
        df = filter_lid_rows(spec["df"], rings, lids).reset_index(drop=True)

        if spec["kind"] == "lids_all":
            columns = [c for c in LID_TABLE_COLUMNS if c in df.columns]
            df = df[columns]
            if df["Ring"].isna().all():
                df = df.drop(columns=["Ring"])
            order = sorted(
                range(len(df)),
                key=lambda i: (
                    int(df["Ring"].iloc[i]) if "Ring" in df.columns else 0,
                    lid_sort_key(df["Lid"].iloc[i]),
                ),
            )
            df = df.iloc[order].reset_index(drop=True)
            self._populate_table(df, float_fmt="{:.6g}",
                                 value_columns=[c for c in LID_VALUE_COLUMNS if c in df.columns])

        elif spec["kind"] == "lids_pivot":
            pivot = build_lid_pivot(df, spec["component"])
            column_labels = [str(c) for c in pivot.columns]
            display = pivot.copy()
            display.columns = column_labels
            max_only = set(column_labels) if is_norm_column(component_column(spec["component"])) else set()
            self._populate_table(display, float_fmt="{:.2f}", value_columns=column_labels,
                                 max_only=max_only, row_labels=[f"Ring {r}" for r in pivot.index])

        else:
            value_columns = [c for c in df.columns if c != "Ring"]
            self._populate_table(df, float_fmt="{}", value_columns=value_columns,
                                 summary=spec.get("summary", False))

    @staticmethod
    def _cell_text(value, float_fmt):
        if pd.isna(value):
            return ""
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            return float_fmt.format(value)
        return str(value)

    def _populate_table(self, df, float_fmt, value_columns, max_only=(), row_labels=None, summary=False):
        """Fill the QTableWidget; red = column maximum, blue = column minimum (norm columns: max only)."""
        n_rows = len(df)
        extra = 1 if summary else 0

        self.table.setRowCount(n_rows + extra)
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])
        if row_labels is not None:
            self.table.setVerticalHeaderLabels(row_labels)

        for i in range(n_rows):
            for j, column in enumerate(df.columns):
                self.table.setItem(i, j, QTableWidgetItem(self._cell_text(df.iloc[i, j], float_fmt)))

        if summary and n_rows:
            for j, column in enumerate(df.columns):
                if column == "Ring":
                    text = "Sum"
                else:
                    text = f"{pd.to_numeric(df[column], errors='coerce').sum():.6g}"
                item = QTableWidgetItem(text)
                item.setBackground(QColor(230, 230, 230))
                self.table.setItem(n_rows, j, item)

        marks = extreme_cells(df, value_columns, max_only)
        columns = list(df.columns)
        for (row, column), mark in marks.items():
            self.table.item(row, columns.index(column)).setBackground(COLOR_MAX if mark == "max" else COLOR_MIN)

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    # =========================================================
    # WORST CASE
    # =========================================================

    def _show_lids_view(self, plot_name):
        self.lids_view_selector.setCurrentText(f"{plot_name}_Lids")

    def _compute_custom_worst_case(self, metric, component, plot_name):
        """Worst case among the Steps for the selected custom study (Lid 1..N, never Lid+Y/...)."""
        study = self.lids_study_selector.currentText()
        lid = None if metric == "PerLid" else metric.replace("PerLid", "", 1).strip()
        steps = [self.step_selector.itemText(i) for i in range(self.step_selector.count())]

        step, value = find_worst_lid_case(self.base_folder, steps, study, component, lid)
        if step is None:
            self.worst_result.setText(f"No hay datos de {study}")
            return

        self.worst_result.setText(f"Worst {metric} → {step} | {component} = {value:.2f}")
        self.step_selector.setCurrentText(step)
        self.lids_study_selector.setCurrentText(study)
        self._show_lids_view(plot_name)

    def compute_worst_case(self):

        metric = self.metric_type.currentText()
        component = self.metric_component.currentText()

        if component.lower() == "normf":
            plot_name = "NormF"
        elif component.lower() == "normt":
            plot_name = "NormT"
        else:
            plot_name = component

        study_kind, _ = self._study_state(os.path.join(self.base_folder, self.step_selector.currentText()))
        if metric.startswith("PerLid") and study_kind == "custom":
            self._compute_custom_worst_case(metric, component, plot_name)
            return

        worst_file = os.path.join(self.base_folder, "WorstCases_Global.txt")

        if not os.path.exists(worst_file):
            self.worst_result.setText("No existe WorstCases_Global.txt")
            return

        df = pd.read_csv(worst_file, sep="\t")

        row = df[df["Component"] == component]

        if row.empty:
            return

        # ===============================
        # PER LID (GLOBAL + SUBDIVISIONES)
        # ===============================
        
        if metric.startswith("PerLid"):
        
            if metric == "PerLid":
                step_col = "PerLid_Step"
                value_col = "PerLid_Value"
            else:
                suffix = metric.replace("PerLid", "")
                step_col = f"PerLid{suffix}_Step"
                value_col = f"PerLid{suffix}_Value"
        
            step = row[step_col].values[0]
            value = row[value_col].values[0]
        
            self.worst_result.setText(
                f"Worst {metric} → {step} | {component} = {value:.2f}"
            )
        
            self.step_selector.setCurrentText(step)
            self.lids_study_selector.setCurrentText(LEGACY_STUDY)
            self._show_lids_view(plot_name)

            return

        elif metric == "SumPerRing":

            step = row["SumPerRing_Step"].values[0]
            value = row["SumPerRing_Value"].values[0]

            self.worst_result.setText(
                f"Worst SumPerRing → {step} | {component} = {value:.2f}"
            )

        elif metric == "PerCube":

            step = row["PerCube_Step"].values[0]
            value = row["PerCube_Value"].values[0]

            self.worst_result.setText(
                f"Worst PerCube → {step} | {component} = {value:.2f}"
            )

        elif metric == "Mounting_Ring":

            step = row["MountingRing_Step"].values[0]
            value = row["MountingRing_Value"].values[0]

            self.worst_result.setText(
                f"Worst Mounting Ring → {step} | {component} = {value:.2f}"
            )

        self.step_selector.setCurrentText(step)
        self.lids_study_selector.setCurrentIndex(0)
        self.plot_selector.setCurrentText(plot_name)


    # =========================================================
    # SYSTEM MANAGEMENT
    # =========================================================

    def on_system_selected(self, index):
        """Called when the system dropdown changes."""
        if self._updating_system:
            return
        if index < 0 or self.system_selector.count() == 0:
            return

        name = self.system_selector.currentText()
        if not name or name == "-- Select System --":
            return

        path = self._systems.get(name, "")

        if not os.path.isdir(path):
            msg = QMessageBox(self)
            msg.setWindowTitle("Path not found")
            msg.setText(
                f"Path not found:\n{path}\n\nWould you like to browse for a new folder?"
            )
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            ret = msg.exec_()

            if ret == QMessageBox.Yes:
                new_path = QFileDialog.getExistingDirectory(
                    self, f"Select folder for '{name}'", ""
                )
                if new_path:
                    self._systems[name] = new_path
                    save_config(self._systems)
                    path = new_path
                else:
                    return
            else:
                return

        self.base_folder = path
        self.folder_display.setText(path)

        # Check that Step* subfolders exist
        steps = [
            f for f in os.listdir(path)
            if os.path.isdir(os.path.join(path, f)) and f.startswith("Step")
        ]

        if not steps:
            QMessageBox.information(
                self,
                "Analysis not found",
                "Required analysis folders not found. Press 'Run Force Assembly Analysis'."
            )
            return

        self.load_steps()

    def on_add_system(self):
        """Prompt for a new system name and folder, then persist it."""
        name, ok = QInputDialog.getText(self, "Add System", "System name:")
        if not ok or not name.strip():
            return
        name = name.strip()

        path = QFileDialog.getExistingDirectory(
            self, f"Select folder for '{name}'", ""
        )
        if not path:
            return

        self._updating_system = True
        self._systems[name] = path
        save_config(self._systems)

        already_present = any(
            self.system_selector.itemText(i) == name
            for i in range(self.system_selector.count())
        )
        if not already_present:
            self.system_selector.addItem(name)

        self._updating_system = False

        QMessageBox.information(
            self,
            "System added",
            f"System '{name}' saved.\nPath: {path}"
        )

    def on_run_analysis(self):
        """Run Force_extractor_assembly.py in the selected system folder."""
        if not self.base_folder:
            QMessageBox.warning(self, "No system", "Please select a system first.")
            return

        script_path = os.path.join(_SCRIPT_DIR, "Force_extractor_assembly.py")
        if not os.path.exists(script_path):
            QMessageBox.critical(
                self,
                "Script not found",
                f"Could not find:\n{script_path}"
            )
            return

        reply = QMessageBox.question(
            self,
            "Run analysis",
            f"Run Force Assembly Analysis for:\n{self.base_folder}?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        lids_reply = QMessageBox.question(
            self,
            "Include Lids",
            "Include Lids?",
            QMessageBox.Yes | QMessageBox.No
        )
        include_lids = (lids_reply == QMessageBox.Yes)

        try:
            cmd = [
                sys.executable, script_path,
                "--base_folder", self.base_folder,
                "--magnet_info", self.system_selector.currentText(),
            ]
            if include_lids:
                cmd.append("--include_lids")

            result = subprocess.run(
                cmd,
                cwd=self.base_folder,
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                QMessageBox.information(self, "Done", "Analysis done.")
                # Reload steps after successful analysis
                self.on_system_selected(self.system_selector.currentIndex())
            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Analysis failed (exit code {result.returncode}).\n\n"
                    + result.stderr[-2000:]
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


# =========================================================


class CustomLidsDialog(QDialog):

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Add Custom Lids")
        self.resize(700, 550)

        base_path = parent.base_folder if parent.base_folder else _SCRIPT_DIR
        self.config_path = build_custom_lid_config_path(base_path)
        self.lines = load_custom_lid_config(self.config_path)

        form_layout = QFormLayout()

        self.y_input = QLineEdit("0")
        self.z_input = QLineEdit("0")
        self.angle_input = QLineEdit("30")
        self.symmetry_y = QCheckBox("Simetría en Y (Symmetry XY)")
        self.symmetry_z = QCheckBox("Simetría en Z (Symmetry XZ)")

        self.angle_input.setToolTip("Ángulo referenciado respecto al eje Z (XZ)")

        point_layout = QHBoxLayout()
        point_layout.addWidget(QLabel("Y:"))
        point_layout.addWidget(self.y_input)
        point_layout.addWidget(QLabel("Z:"))
        point_layout.addWidget(self.z_input)

        angle_label = QLabel("Ángulo referenciado respecto al eje Z (XZ)")
        angle_label.setToolTip("Ángulo referenciado respecto al eje Z (XZ)")

        form_layout.addRow("Punto de paso (Y, Z):", point_layout)
        form_layout.addRow("Ángulo [deg]:", self.angle_input)
        form_layout.addRow(angle_label)
        form_layout.addRow(self.symmetry_y)
        form_layout.addRow(self.symmetry_z)

        self.line_table = QTableWidget(0, 5)
        self.line_table.setHorizontalHeaderLabels([
            "Y step",
            "Z step",
            "Angle",
            "Sym. Y",
            "Sym. Z",
        ])
        self.line_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        buttons_layout = QHBoxLayout()
        self.add_button = QPushButton("Add line")
        self.view_button = QPushButton("View lines")
        self.delete_button = QPushButton("Delete line")
        self.clear_button = QPushButton("Clear lines")
        self.run_button = QPushButton("Run Custom Lids Analysis")

        buttons_layout.addWidget(self.add_button)
        buttons_layout.addWidget(self.view_button)
        buttons_layout.addWidget(self.delete_button)
        buttons_layout.addWidget(self.clear_button)
        buttons_layout.addWidget(self.run_button)

        self.add_button.clicked.connect(self.add_line)
        self.view_button.clicked.connect(self.view_lines)
        self.delete_button.clicked.connect(self.delete_selected_line)
        self.clear_button.clicked.connect(self.clear_lines)
        self.run_button.clicked.connect(self.run_custom_analysis)

        layout = QVBoxLayout(self)
        layout.addLayout(form_layout)
        layout.addLayout(buttons_layout)
        layout.addWidget(self.line_table)

        close_button = QDialogButtonBox(QDialogButtonBox.Close)
        close_button.rejected.connect(self.reject)
        layout.addWidget(close_button)

        self.refresh_line_table()

    def add_line(self):
        try:
            y_value = float(self.y_input.text())
            z_value = float(self.z_input.text())
            angle_value = float(self.angle_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid value", "Check the values of Y, Z and angle.")
            return

        line = {
            "y_step": y_value,
            "z_step": z_value,
            "angle_deg": angle_value,
            "symmetry_y": self.symmetry_y.isChecked(),
            "symmetry_z": self.symmetry_z.isChecked(),
        }

        self.lines.append(line)
        self.save_lines()
        self.refresh_line_table()

    def save_lines(self):
        if not self.parent.base_folder:
            return
        save_custom_lid_config(self.config_path, self.lines)

    def refresh_line_table(self):
        self.line_table.setRowCount(len(self.lines))
        for i, line in enumerate(self.lines):
            self.line_table.setItem(i, 0, QTableWidgetItem(f"{float(line['y_step']):.3f}"))
            self.line_table.setItem(i, 1, QTableWidgetItem(f"{float(line['z_step']):.3f}"))
            self.line_table.setItem(i, 2, QTableWidgetItem(f"{float(line['angle_deg']):.2f}"))
            self.line_table.setItem(i, 3, QTableWidgetItem("Yes" if line.get("symmetry_y", False) else "No"))
            self.line_table.setItem(i, 4, QTableWidgetItem("Yes" if line.get("symmetry_z", False) else "No"))

    def clear_lines(self):
        self.lines = []
        self.save_lines()
        self.refresh_line_table()

    def delete_selected_line(self):
        row = self.line_table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "No line selected", "Select a row first.")
            return
        del self.lines[row]
        self.save_lines()
        self.refresh_line_table()

    def view_lines(self):
        if not self.parent.base_folder:
            QMessageBox.warning(self, "No system", "Select a system first.")
            return

        if not self.lines:
            QMessageBox.warning(self, "No lines", "Add at least one divider line before previewing.")
            return

        step_name = self.parent.step_selector.currentText()
        if not step_name:
            QMessageBox.warning(self, "No step", "Select a Step folder first.")
            return

        step_path = os.path.join(self.parent.base_folder, step_name)

        # La vista previa vive en un directorio temporal para no ensuciar la carpeta del Step
        with tempfile.TemporaryDirectory() as tmp_dir:
            preview_path = os.path.join(tmp_dir, "CustomLids_preview.png")
            try:
                make_preview_png(step_path, self.lines, output_path=preview_path)
            finally:
                plt.close("all")
            pixmap = QPixmap(preview_path) if os.path.exists(preview_path) else QPixmap()

        if pixmap.isNull():
            QMessageBox.warning(self, "Preview unavailable", "The custom-lid preview could not be generated.")
            return

        preview_dialog = QDialog(self)
        preview_dialog.setWindowTitle("Custom Lids Preview")
        preview_dialog.resize(900, 900)
        label = QLabel(preview_dialog)
        label.setPixmap(pixmap.scaled(850, 850, Qt.KeepAspectRatio))
        layout = QVBoxLayout(preview_dialog)
        layout.addWidget(label)
        preview_dialog.exec_()

    def run_custom_analysis(self):
        if not self.parent.base_folder:
            QMessageBox.warning(self, "No system", "Select a system first.")
            return

        if not self.lines:
            QMessageBox.warning(self, "No lines", "Add at least one divider line to run the custom lid analysis.")
            return

        self.save_lines()
        script_path = os.path.join(_SCRIPT_DIR, "Force_extractor_assembly.py")
        cmd = [
            sys.executable,
            script_path,
            "--base_folder",
            self.parent.base_folder,
            "--magnet_info",
            self.parent.system_selector.currentText(),
            "--custom_lids_config",
            self.config_path,
        ]

        try:
            result = subprocess.run(cmd, cwd=self.parent.base_folder, capture_output=True, text=True)
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return

        if result.returncode == 0:
            self.parent.load_steps()
            self.parent.refresh_lids_study_selector()
            self.parent.update_view()
            QMessageBox.information(self, "Custom Lids analysis done", "The custom lid study was generated successfully.")
        else:
            QMessageBox.critical(self, "Error", f"Custom lids analysis failed.\n\n{result.stderr[-2000:]}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = StepViewer()
    viewer.show()
    sys.exit(app.exec_())