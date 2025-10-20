#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/10/19 22:18
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   predictor.py
# @Desc     :   

from numpy import ndarray, exp
from os import path
from pandas import DataFrame
from pprint import pprint
from torch import load, device, no_grad

from utils.config import (MODEL_SAVE_PATH, TEST_DATASET,
                          ACCELERATOR,
                          HIDDEN_UNITS)
from utils.models import TorchLinearModel
from utils.stats import load_data_for_test, summary_data
from utils.PT import df2tensor
from main import data_preparation


def main() -> None:
    """ Main Function """
    # Check if model file exists
    if path.exists(MODEL_SAVE_PATH):
        print(f"Model file found at: {MODEL_SAVE_PATH}")

        # Prepare data
        X_raw = load_data_for_test(TEST_DATASET)
        summary_data(X_raw)

        # Get preprocessor and important features list
        _, _, preprocessor, important_features = data_preparation()

        # Process data
        out = preprocessor.transform(X_raw)
        # If the processed data is a sparse matrix, convert it to a dense array
        if hasattr(out, "toarray"):
            out: ndarray = out.toarray()
        X_test: DataFrame = DataFrame(data=out, columns=preprocessor.get_feature_names_out())
        summary_data(X_test)

        # Get important data
        X_test_pca = X_test[important_features]
        summary_data(X_test_pca)
        # Transform to tensor
        X_tensor = df2tensor(X_test_pca, ACCELERATOR)

        # Due to the saved model structure, we need to define the model architecture again
        features: int = X_test_pca.shape[1]
        model = TorchLinearModel(features, HIDDEN_UNITS, 1)
        state_dict = load(MODEL_SAVE_PATH, map_location=device(ACCELERATOR))
        model.load_state_dict(state_dict)
        model.eval()
        print("Model loaded successfully.")

        # Make predictions
        with no_grad():
            pred_predictions = model(X_tensor).cpu().numpy()
            print(f"Predictions shape: {pred_predictions.shape}")
            print(f"First 10 predictions:")
            pprint(pred_predictions[:10])

    else:
        print(f"Model file not found at: {MODEL_SAVE_PATH}. Please train the model first.")


if __name__ == "__main__":
    main()
