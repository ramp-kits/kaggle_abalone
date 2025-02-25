import os
from typing import Optional, Tuple
import json
import numpy as np
import pandas as pd
import ramphy.ramp_setup as rs
from ramphy import Hyperparameter

# RAMP START HYPERPARAMETERS
_Sex_1_to_drop = Hyperparameter(
    dtype='bool', default=False, values=[False, True, ])
_Length_2_to_drop = Hyperparameter(
    dtype='bool', default=False, values=[False, True, ])
_Diameter_3_to_drop = Hyperparameter(
    dtype='bool', default=False, values=[False, True, ])
_Height_4_to_drop = Hyperparameter(
    dtype='bool', default=False, values=[False, True, ])
_Whole_weight_5_to_drop = Hyperparameter(
    dtype='bool', default=False, values=[False, True, ])
_Whole_weight_1_6_to_drop = Hyperparameter(
    dtype='bool', default=False, values=[False, True, ])
_Whole_weight_2_7_to_drop = Hyperparameter(
    dtype='bool', default=False, values=[False, True, ])
_Shell_weight_8_to_drop = Hyperparameter(
    dtype='bool', default=False, values=[False, True, ])
# RAMP END HYPERPARAMETERS

class DataPreprocessor(rs.BaseDataPreprocessor):
    """Drops columns."""

    def preprocess(
        self, X_train: pd.DataFrame, y_train: np.ndarray, X_test: pd.DataFrame, metadata: dict
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict]:
        cols_to_drop = []
        for col in X_train.columns:
            if bool(eval(f"{col}_to_drop")):
                cols_to_drop.append(col)
        print(f"Dropping {cols_to_drop}")
        X_train = X_train.drop(cols_to_drop, axis=1)
        X_test = X_test.drop(cols_to_drop, axis=1)
        for col in cols_to_drop:
            metadata["data_description"]["feature_types"].pop(col)
        return X_train, y_train, X_test, metadata
