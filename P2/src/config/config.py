from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="APP",
    settings_files=["config/settings.toml"],
)

class Config:
    permission_file = settings.dataset_paths.permission_file
    api_file = settings.dataset_paths.api_file
    features_csv = settings.dataset_paths.features_csv
    drebin_csv = settings.dataset_paths.drebin_csv
    test_size = settings.train_test_split.test_size
    apks_path = settings.androguard_paths.apk_path
    knn_model1_path = settings.knn.model1_path
    knn_model2_path = settings.knn.model2_path
    dt_model1_path = settings.dt.model1_path
    dt_model2_path = settings.dt.model2_path

config = Config()
