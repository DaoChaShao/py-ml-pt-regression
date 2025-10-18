#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/19 00:09
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   models.py
# @Desc     :   

from torch import nn, Tensor


class Model(nn.Module):
    """ A simple feedforward neural network model """

    def __init__(self, features: int, hidden_units: int, output_size: int) -> None:
        """ Initialise the Model class
        :param features: number of input features
        :param hidden_units: number of hidden units
        :param output_size: number of output classes
        """
        super().__init__()
        self._input = nn.Linear(features, hidden_units)
        self._hidden = nn.Sequential(
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self._output = nn.Linear(hidden_units, output_size)

    def forward(self, features: Tensor) -> Tensor:
        """ Forward pass of the model
        :param features: input tensor
        :return: output tensor
        """
        out = self._input(features)
        out = self._hidden(out)
        out = self._output(out)
        return out
