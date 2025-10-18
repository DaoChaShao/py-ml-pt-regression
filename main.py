#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/18 12:00
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   home.py
# @Desc     :   

from utils.config import TRAIN_DATASET
from utils.helper import Timer
from utils.stats import (load_data, summary_data,
                         preprocess_data,
                         pca_importance,)
from utils.PT import TorchDataset


def main() -> None:
    """ Main Function """
    # Load training raw dataset
    X_train_raw, y_train_raw = load_data(TRAIN_DATASET)

    # Summary of raw data
    summary_data(X_train_raw)
    summary_data(y_train_raw)

    # Preprocess the raw data
    X_train, preprocessor = preprocess_data(X_train_raw)

    # Summary of preprocessed data
    summary_data(X_train)

    # Get the important features using PCA
    important_features, pca_model, _ = pca_importance(X_train)
    # # Get the important data
    X_train_pca = X_train[important_features]
    # print(X_train_pca)
    # print(y_train_raw)

    trainer = TorchDataset(X_train_pca, y_train_raw)
    print(trainer)


if __name__ == "__main__":
    main()
