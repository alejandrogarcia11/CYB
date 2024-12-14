from androguard.util import set_log
from src.config.config import config
from src.data_classes.apk_model_evaluator import APKModelEvaluator
from src.data_classes.dataset import Dataset
from src.data_classes.model import Model
from src.data_classes.trainer import Trainer

if __name__ == "__main__":
    set_log("ERROR")

    dataset = Dataset()

    # Train and evaluate the K-Nearest Neighbors and Decision Tree models
    print("KNN model:\n")

    knn_model = Model(model_type="knn")
    knn_trainer = Trainer(dataset=dataset, model=knn_model)
    knn_trainer.train_and_evaluate()

    print("-------------------------------------------------")
    print("\nDT model:\n")

    dt_model = Model(model_type="decision_tree")
    dt_trainer = Trainer(dataset=dataset, model=dt_model)
    dt_trainer.train_and_evaluate()

    # Evaluate the models using APK files
    apk_folder = config.apks_path
    model_paths_knn = {"knn_1": config.knn_model1_path,
                       "knn_2": config.knn_model2_path}

    model_paths_dt = {"decision_tree_1": config.dt_model1_path,
                      "decision_tree_2": config.dt_model2_path}

    knn_evaluator = APKModelEvaluator(model_paths=model_paths_knn,
                                      apk_folder=apk_folder,
                                      permissions_features=dataset.permissions_features,
                                      api_features=dataset.api_features)
    knn_evaluator.run()

    dt_evaluator = APKModelEvaluator(model_paths=model_paths_dt,
                                     apk_folder=apk_folder,
                                     permissions_features=dataset.permissions_features,
                                     api_features=dataset.api_features)
    dt_evaluator.run()
