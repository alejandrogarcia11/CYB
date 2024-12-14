import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

from src.config.config import settings


class Model:
    def __init__(self, model_type: str = "knn"):
        """
        Initializes the model based on the specified type.
        :param model_type: Type of model to be used (knn or decision_tree).
        """
        if model_type == "knn":
            self.model = KNeighborsClassifier()
        elif model_type == "decision_tree":
            self.model = DecisionTreeClassifier()
        else:
            raise ValueError(f"Model type {model_type} is not supported.")

        self.model_type = model_type

    def train(self,
              x_train: pd.DataFrame,
              y_train: pd.Series,
              model_path: str) -> None:
        """
        Trains the model using the training data and saves it to a .pkl file.
        :param x_train: Training features
        :param y_train: Training labels
        :param model_path: Path to save the trained model
        :return:
        """
        self.model.fit(x_train, y_train)

        # Save the trained model to the path specified in the config file
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[float, float, float, float]:
        """
        Evaluates the model using the test data.
        :param x_test: Test features
        :param y_test: Test labels
        :return: Tuple containing accuracy, precision, recall, and F1 score
        """
        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="binary", pos_label=1)
        recall = recall_score(y_test, y_pred, average="binary", pos_label=1)
        f1 = f1_score(y_test, y_pred, average="binary", pos_label=1)

        return accuracy, precision, recall, f1