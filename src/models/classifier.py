from abc import ABC, abstractmethod
from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class Classifier(ABC):
    @abstractmethod
    def train(self, *params) -> None:
        pass

    @abstractmethod
    def evaluate(self, *params) -> Dict[str, float]:
        pass

    @abstractmethod
    def predict(self, *params) -> np.ndarray:
        pass


class SklearnClassifier(Classifier):
    def __init__(
        self, estimator: BaseEstimator, features: List[str], target: str,
    ):
        self.clf = estimator
        self.features = features
        self.target = target

    def train(self, df_train: pd.DataFrame):
        self.clf.fit(df_train[self.features].values, df_train[self.target].values)

    def predict(self, df: pd.DataFrame):
        return self.clf.predict_proba(df[self.features].values)[:, 1]

    def evaluate(self, df_test: pd.DataFrame):
        y_pred = self.clf.predict(df_test[self.features].values)
        y_orig = df_test[self.target].values
        accuracy = accuracy_score(y_orig, y_pred)
        precision = precision_score(y_orig, y_pred)
        recall = recall_score(y_orig, y_pred)
        f1 = f1_score(y_orig, y_pred)
        output = {'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1}
        return (output)


        # raise NotImplementedError(
        #     f"You're almost there! Identify an appropriate evaluation metric for your model and implement it here. "
        #     f"The expected output is a dictionary of the following schema: {{metric_name: metric_score}}"
        # )

    
