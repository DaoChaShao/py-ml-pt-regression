#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/18 13:00
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   config.py
# @Desc     :   

# Data file paths
TRAIN_DATASET: str = "data/train.csv"
TEST_DATASET: str = "data/test.csv"

# Model save path
MODEL_SAVE_PATH: str = "models/model.pth"

# Model hyperparameters
ALPHA: float = 0.01
ALPHA4REDUCTION: float = 0.3
BATCH_SIZE: int = 32
EPOCHS: int = 100
