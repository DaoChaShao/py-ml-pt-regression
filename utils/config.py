#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/18 13:00
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   config.py
# @Desc     :   

from pathlib import Path

# Set base directory
BASE_DIRECTORY = Path(__file__).resolve().parent.parent

# Data file paths
TRAIN_DATASET = BASE_DIRECTORY / "data/train.csv"
TEST_DATASET = BASE_DIRECTORY / "data/test.csv"

# Model save path
MODEL_SAVE_PATH = BASE_DIRECTORY / "models/model.pth"

# Data processing parameters
RANDOM_STATE: int = 27
VALID_SIZE: float = 0.2
IS_SHUFFLE: bool = True

# PCA parameters
PCA_VARIANCE_THRESHOLD: float = 0.95

# Data settings
BATCHES: int = 32

# Training hyperparameters
HIDDEN_UNITS: int = 128
ALPHA: float = 0.01
ALPHA4REDUCTION: float = 0.3
EPOCHS: int = 100
ACCELERATOR: str = "cpu"
