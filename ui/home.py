#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/18 15:03
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   home.py
# @Desc     :   

from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtCore import QSortFilterProxyModel
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout,
                               QPushButton, QTabWidget,
                               QTableView, QHeaderView)
from sys import argv, exit

from utils.config import TRAIN_DATASET
from utils.stats import load_data


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Regression Prediction Tool")
        self.resize(1200, 800)
        self._widget = QWidget(self)
        self.setCentralWidget(self._widget)

        self._func_labels = ["Load", "Exit"]

        self._tab = QTabWidget(self)

        self._model_train = QStandardItemModel(self)
        self._agent_train = QSortFilterProxyModel(self._model_train)
        self._agent_train.setSourceModel(self._model_train)
        self._table_train = QTableView(self)
        self._table_train.setModel(self._agent_train)

        self._setup()

    def _setup(self) -> None:
        _layout = QVBoxLayout(self._widget)
        _row_btn = QHBoxLayout()

        # Create Tab for Train Data
        _tab_train = QWidget()
        _row_train = QVBoxLayout(_tab_train)
        _row_train.addWidget(self._table_train)
        self._tab.addTab(_tab_train, "Train Data")

        _layout.addWidget(self._tab)

        funcs = [
            self._load,
            self._exit,
        ]
        for i, label in enumerate(self._func_labels):
            button = QPushButton(label, self)
            button.clicked.connect(funcs[i])
            _row_btn.addWidget(button)
        _layout.addLayout(_row_btn)

        self._widget.setLayout(_layout)

        # Setup Train Table
        self._table_train.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table_train.setSortingEnabled(True)

    def _exit(self):
        self.close()

    def _load(self):
        X, y = load_data(f"../{TRAIN_DATASET}")

        # Clear the previous data
        self._model_train.clear()

        # Set headers
        headers: list[str] = X.columns.tolist()
        for i, header in enumerate(headers[:10]):
            self._model_train.setHorizontalHeaderItem(i, QStandardItem(header))

        # Populate data
        for i, row in X.iloc[:, :10].iterrows():
            items = [QStandardItem(str(row[header])) for header in headers[:10]]
            self._model_train.appendRow(items)


if __name__ == "__main__":
    app = QApplication(argv)
    window = MainWindow()
    window.show()
    exit(app.exec())
