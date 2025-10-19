#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/19 18:15
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   plot.py
# @Desc     :   

from PySide6.QtCharts import QLineSeries, QChart, QChartView
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout,
                               QPushButton, )
from sys import argv, exit

from main import main


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

        self._line_titles: list[str] = ["Train Loss", "Valid Loss"]
        self._train_losses: list = []
        self._valid_losses: list = []

        self._setup()

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
        # Delete previous series
        self._chart.removeAllSeries()

        # Start to train
        main()

        for i, line in enumerate(self._line_titles):
            series = QLineSeries()
            series.setName(line)
            data = self._line_data[i]
            for j, point in enumerate(data):
                series.append(j, point)
            self._chart.addSeries(series)

        self._chart.setTitle(" & ".join(self._line_titles))
        self._chart.createDefaultAxes()

        self._chart.update()

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

    def get_train_loss(self, epoch: int, loss: float) -> None:
        self._train_losses.append((epoch, loss))

    def get_valid_loss(self, epoch: int, loss: float) -> None:
        self._valid_losses.append((epoch, loss))


if __name__ == "__main__":
    app = QApplication(argv)
    window = MainWindow()
    window.show()
    exit(app.exec())
