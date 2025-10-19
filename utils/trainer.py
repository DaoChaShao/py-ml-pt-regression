#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/19 01:11
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   trainer.py
# @Desc     :   

from PySide6.QtCore import QObject, Signal
from torch import nn, no_grad, save, device
from torch.utils.data import DataLoader

from utils.PT import get_device, TorchDataLoader, log_mse_loss


class TorchTrainer(QObject):
    """ Trainer class for managing training process """
    losses: Signal = Signal(int, float, float)

    def __init__(self, model: nn.Module, optimiser, accelerator: str = "auto") -> None:
        super().__init__()
        """ Initialise the Trainer class
        :param model: the neural network model to be trained
        :param optimiser: the optimiser for updating model parameters
        :param accelerator: device to use for training ("cpu", "cuda", or "auto
        """
        self._model = model
        self._optimiser = optimiser
        self._accelerator = get_device(accelerator)

        self._train_losses: list[float] = []
        self._valid_losses: list[float] = []

    def _epoch_train(self, dataloader: DataLoader | TorchDataLoader) -> float:
        """ Train the model for one epoch
        :param dataloader: DataLoader for training data
        :return: average training loss for the epoch
        """
        # Set model to training mode
        self._model.train()

        _loss: float = 0.0
        for features, labels in dataloader:
            features, labels = features.to(device(self._accelerator)), labels.to(device(self._accelerator))

            self._optimiser.zero_grad()
            outputs = self._model(features)

            loss = log_mse_loss(outputs.squeeze(), labels)
            loss.backward()
            self._optimiser.step()

            _loss += loss.item() * features.size(0)

        return _loss / len(dataloader.dataset)

    def _epoch_valid(self, dataloader: DataLoader | TorchDataLoader) -> float:
        """ Validate the model for one epoch
        :param dataloader: DataLoader for validation data
        :return: average validation loss for the epoch
        """
        # Set model to evaluation mode
        self._model.eval()

        _loss: float = 0.0
        with no_grad():
            for features, labels in dataloader:
                features, labels = features.to(device(self._accelerator)), labels.to(device(self._accelerator))

                outputs = self._model(features)
                loss = log_mse_loss(outputs.squeeze(), labels)

                _loss += loss.item() * features.size(0)

        return _loss / len(dataloader.dataset)

    def fit(self,
            train_loader: DataLoader | TorchDataLoader, valid_loader: DataLoader | TorchDataLoader,
            epochs: int, model_save_path: str | None = None) -> None:
        """ Fit the model to the training data
        :param train_loader: DataLoader for training data
        :param valid_loader: DataLoader for validation data
        :param epochs: number of training epochs
        :param model_save_path: path to save the best model parameters
        :return: None
        """
        _best_valid_loss = float("inf")

        for epoch in range(epochs):
            train_loss = self._epoch_train(train_loader)
            valid_loss = self._epoch_valid(valid_loader)

            self._train_losses.append(train_loss)
            self._valid_losses.append(valid_loss)
            # Emit training and validation progress signal
            self.losses.emit(epoch + 1, train_loss, valid_loss)

            print(f"Epoch [{epoch + 1}/{epochs}] - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Valid Loss: {valid_loss:.4f}")

            # Save the model if it has the best validation loss so far
            if valid_loss < _best_valid_loss:
                _best_valid_loss = valid_loss
                save(self._model.state_dict(), model_save_path)
                print(f"Model's parameters saved to {model_save_path}")
