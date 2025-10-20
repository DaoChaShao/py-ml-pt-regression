#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/19 18:15
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   plot.py
# @Desc     :   

from PySide6.QtCore import Signal, QThread, Qt
from PySide6.QtCharts import QLineSeries, QChart, QChartView
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout,
                               QPushButton, )
from sys import argv, exit

from torch import optim, device

from main import data_preparation
from utils.config import (HIDDEN_UNITS, ACCELERATOR,
                          ALPHA,
                          EPOCHS, MODEL_SAVE_PATH)
from utils.models import TorchLinearModel
from utils.trainer import TorchTrainer


class TrainerThread(QThread):
    losses = Signal(int, float, float)

    def run(self):
        train_loader, valid_loader, _, _ = data_preparation()
        model = TorchLinearModel(train_loader[0][0].shape[0], HIDDEN_UNITS, 1)
        model.to(device(ACCELERATOR))
        optimiser = optim.Adam(model.parameters(), lr=ALPHA)
        trainer = TorchTrainer(model, optimiser, ACCELERATOR)

        trainer.losses.connect(self.losses)

        trainer.fit(train_loader, valid_loader, EPOCHS, MODEL_SAVE_PATH)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optimiser with Adam Learning Rate Scheduler")
        self.resize(800, 400)
        self._widget = QWidget(self)
        self.setCentralWidget(self._widget)

        self._chart = QChart()
        self._view = QChartView(self._chart)
        self._btn_labels = ["Plot", "Clear", "Exit"]
        self._buttons = []

        self._series_train_loss = QLineSeries()
        self._series_valid_loss = QLineSeries()

        self._setup()

        self._thread = TrainerThread()
        self._thread.losses.connect(self._get_losses)

    def _setup(self):
        _layout = QVBoxLayout()
        _row = QHBoxLayout()

        # Chart View
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        _layout.addWidget(self._view)

        funcs = [
            self._click2plot,
            self._click2clear,
            self.close,
        ]
        for i, label in enumerate(self._btn_labels):
            button = QPushButton(label, self)
            button.clicked.connect(funcs[i])
            if button.text() == "Clear":
                button.setEnabled(False)
            self._buttons.append(button)
            _row.addWidget(button)
        _layout.addLayout(_row)

        self._widget.setLayout(_layout)

    def _click2plot(self) -> None:
        """ Plot random data points """
        # Initialise the chart and data docker
        self._chart.removeAllSeries()
        self._series_train_loss.clear()
        self._series_valid_loss.clear()

        self._series_train_loss.setName("Train Loss")
        self._series_valid_loss.setName("Valid Loss")
        self._chart.addSeries(self._series_train_loss)
        self._chart.addSeries(self._series_valid_loss)
        self._chart.createDefaultAxes()
        self._chart.setTitle("Training and Validation Loss over Epochs")

        # Start to train in a separate thread
        self._thread.start()

        for button in self._buttons:
            if button.text() == "Clear":
                button.setEnabled(True)
        for button in self._buttons:
            if button.text() == "Plot":
                button.setEnabled(False)

    def _click2clear(self) -> None:
        """ Clear the chart """
        self._chart.setTitle("")
        self._chart.removeAllSeries()
        for axis in self._chart.axes():
            self._chart.removeAxis(axis)

        for button in self._buttons:
            if button.text() == "Clear":
                button.setEnabled(False)
        for button in self._buttons:
            if button.text() == "Plot":
                button.setEnabled(True)

    def _get_losses(self, epoch: int, loss_train: float, loss_valid: float) -> None:
        # print(f"[Signal] Epoch {epoch}: Train Loss {loss_train:.4f} - Valid Loss {loss_valid:.4f}")
        self._series_train_loss.append(epoch, loss_train)
        self._series_valid_loss.append(epoch, loss_valid)

        # Set axis x range dynamically
        axis_x = self._chart.axes(Qt.Orientation.Horizontal, self._series_train_loss)[0]
        axis_x.setRange(0, epoch * 1.1)

        # Set axis y range dynamically
        axis_y = self._chart.axes(Qt.Orientation.Vertical, self._series_train_loss)[0]
        points_y = [self._series_train_loss.at(i).y() for i in range(self._series_train_loss.count())]
        if points_y:
            min_y = min(points_y)
            max_y = max(points_y)
            axis_y.setRange(min_y * 0.9, max_y * 1.1)

        # Update the view
        self._view.update()


if __name__ == "__main__":
    app = QApplication(argv)
    window = MainWindow()
    window.show()
    exit(app.exec())
