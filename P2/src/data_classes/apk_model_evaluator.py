import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.data_classes.androguard import AndroguardProcessor


class APKModelEvaluator:
    def __init__(self,
                 model_paths: dict,
                 apk_folder: str,
                 permissions_features: list,
                 api_features: list):
        """
        Initializes the evaluator with the paths to the trained models and the folder containing APKs.

        :param model_paths: Dictionary with the paths to the trained models (e.g., {'knn': 'knn_model.pkl', 'decision_tree': 'dt_model.pkl'}).
        :param apk_folder: Path to the folder containing the APK files to evaluate.
        """
        self.model_paths = model_paths
        self.apk_folder = apk_folder
        self.permissions_features = permissions_features
        self.api_features = api_features
        self.models = self.load_models()

        # DataFrames for features
        self.manifest_features_df = pd.DataFrame()
        self.full_features_df = pd.DataFrame()

    def load_models(self) -> dict:
        """
        Loads the trained models from the specified paths.

        :return: Dictionary containing the loaded models.
        """
        models = {}
        for model_name, model_path in self.model_paths.items():
            models[model_name] = joblib.load(model_path)
        return models

    def extract_features_from_apk(self, apk_path: str, label: int) -> None:
        """
        Extracts features from a given APK file and appends them to the appropriate DataFrame.

        :param apk_path: Path to the APK file.
        :param label: Label for the APK (e.g., 1 for malware, 0 for benign).
        """
        processor = AndroguardProcessor(apk_path)
        processor.load_apk()
        permissions = processor.get_permissions()
        api_calls = processor.get_api_calls()

        # Encode and append features to the DataFrames
        manifest_features = self.encode_features(permissions, self.permissions_features)
        full_features = self.encode_features(api_calls + permissions, self.permissions_features + self.api_features)

        # Add label and APK name to both feature rows
        manifest_features['label'] = label
        manifest_features['apk_name'] = os.path.basename(apk_path)
        full_features['label'] = label
        full_features['apk_name'] = os.path.basename(apk_path)

        # Append to corresponding DataFrames
        self.manifest_features_df = pd.concat([self.manifest_features_df, manifest_features], ignore_index=True)
        self.full_features_df = pd.concat([self.full_features_df, full_features], ignore_index=True)

    def encode_features(self,
                        features: list,
                        features_template: list) -> pd.DataFrame:
        """
        Encodes the features (permissions or API calls) into a one-hot encoded DataFrame row.

        :param features: List of features (either permissions or API calls) extracted from the APK.
        :param features_template: List of all possible features (permissions or API calls).
        :return: DataFrame with one-hot encoded features as a single row.
        """
        # Dict with all used features set to 0
        feature_dict = {feature: 0 for feature in features_template}

        # Set to 1 if the feature is present in the list of extracted features
        feature_dict.update({feature: 1 for feature in features if feature in feature_dict})

        return pd.DataFrame([feature_dict])

    def process_all_apks(self) -> None:
        """
        Processes all APKs in the specified folder, extracts their features, and stores them in the DataFrames.
        """
        for subfolder, label in [('malware-samples', 1), ('vulnerable-samples', 0)]:
            folder_path = os.path.join(self.apk_folder, subfolder)
            for apk_file in os.listdir(folder_path):
                print(f"Processing APK: {apk_file}")
                apk_path = os.path.join(folder_path, apk_file)
                self.extract_features_from_apk(apk_path, label)

    def evaluate_model_on_features(self, model, features, labels, feature_type: str) -> dict:
        """
        Evaluates a model on a given set of features and calculates performance metrics.

        :param model: Trained model to evaluate.
        :param features: Features to evaluate the model on.
        :param labels: True labels for the features.
        :param feature_type: Type of features ('manifest' or 'full').
        :return: Dictionary with evaluation results.
        """
        # Get predictions
        predictions = model.predict(features)

        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        conf_matrix = confusion_matrix(labels, predictions)

        # Store metrics
        results = {
            'model': model,
            f'{feature_type}_accuracy': accuracy,
            f'{feature_type}_precision': precision,
            f'{feature_type}_recall': recall,
            f'{feature_type}_f1_score': f1,
            f'{feature_type}_confusion_matrix': conf_matrix.tolist()
        }

        return results

    def evaluate_models(self) -> None:
        """
        Evaluates all models using the extracted features, calculates metrics, and saves the results.
        """
        # Prepare the features and labels for both datasets
        manifest_features = self.manifest_features_df.drop(columns=['label', 'apk_name'])
        manifest_labels = self.manifest_features_df['label']

        full_features = self.full_features_df.drop(columns=['label', 'apk_name'])
        full_labels = self.full_features_df['label']

        metrics_results = []

        for model_name, model in self.models.items():
            # Determine the features the model expects
            model_feature_names = model.feature_names_in_

            # Check if model expects only manifest features
            if set(model_feature_names) == set(self.permissions_features):
                # Evaluate model on manifest features
                manifest_results = self.evaluate_model_on_features(model, manifest_features, manifest_labels,
                                                                   'manifest')
                manifest_results['model'] = model_name
                manifest_results['feature_set'] = 'manifest'
                metrics_results.append(manifest_results)

            # Check if model expects the full set of features
            if set(model_feature_names) == set(self.permissions_features + self.api_features):
                # Evaluate model on full features
                full_results = self.evaluate_model_on_features(model, full_features, full_labels, 'full')
                full_results['model'] = model_name
                full_results['feature_set'] = 'full'
                metrics_results.append(full_results)

        print(metrics_results)

    def run(self) -> None:
        """
        Runs the entire process: feature extraction and model evaluation.
        """
        print("Extracting features from APKs...")
        self.process_all_apks()
        print("Features extracted. Evaluating models...")
        self.evaluate_models()
        print("Evaluation complete.")
