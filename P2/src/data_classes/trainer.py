from src.config.config import config


class Trainer:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
        if model.model_type == "knn":
            self.manifest_only, self.api_manifest = config.knn_model1_path, config.knn_model2_path
        else:
            self.manifest_only, self.api_manifest = config.dt_model1_path, config.dt_model2_path

    def train_and_evaluate(self) -> None:
        """
        Trains and evaluates the model using both Manifest Permissions and API call features.

        :return:
        """

        X_train_perm, X_test_perm, X_train_all, X_test_all, y_train, y_test = self.dataset.split_data()

        print(f"Training and evaluating model using only Manifest Permissions features:")
        self.model.train(X_train_perm, y_train, self.manifest_only)
        accuracy_perm, precision_perm, recall_perm, f1_perm = self.model.evaluate(X_test_perm, y_test)
        print(
            f"Accuracy: {accuracy_perm:.4f}, Precision: {precision_perm:.4f}, Recall: {recall_perm:.4f}, F1-score: {f1_perm:.4f}")

        print(f"\nTraining and evaluating model using both Manifest Permissions and API call features:")
        self.model.train(X_train_all, y_train, self.api_manifest)
        accuracy_all, precision_all, recall_all, f1_all = self.model.evaluate(X_test_all, y_test)
        print(
            f"Accuracy: {accuracy_all:.4f}, Precision: {precision_all:.4f}, Recall: {recall_all:.4f}, F1-score: {f1_all:.4f}")
