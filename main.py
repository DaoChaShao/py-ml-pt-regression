#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/18 12:00
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   main.py
# @Desc     :

from torch import optim, device

from utils.config import (TRAIN_DATASET,
                          VALID_SIZE, RANDOM_STATE, IS_SHUFFLE,
                          PCA_VARIANCE_THRESHOLD,
                          BATCHES,
                          HIDDEN_UNITS,
                          ACCELERATOR,
                          ALPHA,
                          EPOCHS, MODEL_SAVE_PATH)
from utils.helper import Timer
from utils.models import TorchLinearModel
from utils.PT import TorchDataset, TorchDataLoader, TorchRandomSeed
from utils.stats import (load_data, summary_data,
                         preprocess_data,
                         split_data,
                         pca_importance, )
from utils.trainer import TorchTrainer


def data_preparation() -> tuple[TorchDataLoader, TorchDataLoader, list]:
    """ Data Preparation """
    # Load training raw dataset
    X_train_raw, y_train_raw = load_data(TRAIN_DATASET)

    # Summary of raw data
    summary_data(X_train_raw)
    summary_data(y_train_raw)

    # Preprocess the raw data
    X_train, preprocessor = preprocess_data(X_train_raw)

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = split_data(X_train, y_train_raw, VALID_SIZE, RANDOM_STATE, IS_SHUFFLE)

    # Summary of preprocessed data
    summary_data(X_train)
    summary_data(X_valid)

    # Get the important features using PCA
    important_features, _, _ = pca_importance(X_train, PCA_VARIANCE_THRESHOLD)
    # Get the important data
    X_train_pca = X_train[important_features]
    X_valid_pca = X_valid[important_features]

    # Create Torch Datasets
    train_set = TorchDataset(X_train_pca, y_train_raw)
    valid_set = TorchDataset(X_valid_pca, y_valid)

    # Create Torch DataLoaders
    train_loader = TorchDataLoader(train_set, BATCHES, IS_SHUFFLE)
    valid_loader = TorchDataLoader(valid_set, BATCHES, IS_SHUFFLE)

    return train_loader, valid_loader, important_features


def main() -> None:
    """ Main Function """
    with Timer("Data Preparation"):
        train_loader, valid_loader, _ = data_preparation()
        # print(train_loader)
        # print(train_loader[0])

    with TorchRandomSeed("Training", RANDOM_STATE):
        features_size: int = train_loader[0][0].shape[0]
        # Set up and train the model
        model = TorchLinearModel(
            features=features_size,
            hidden_units=HIDDEN_UNITS,
            output_size=1
        )
        model.to(device(ACCELERATOR))

        # Set up optimiser
        optimiser = optim.Adam(model.parameters(), lr=ALPHA)

        # Set up the trainer
        trainer = TorchTrainer(model, optimiser, ACCELERATOR)
        trainer.fit(train_loader, valid_loader, EPOCHS, MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
