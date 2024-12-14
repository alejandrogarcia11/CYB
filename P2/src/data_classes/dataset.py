import pandas as pd
from sklearn.model_selection import train_test_split
from src.config.config import config

class Dataset:
    def __init__(self):
        self.labels = None
        self.permissions_features = None
        self.api_features = None
        self.drebin_df = None
        self.load_data()

    def load_data(self) -> None:
        """
        Loads and preprocesses the Drebin dataset.
        """
        drebin_df = pd.read_csv(config.drebin_csv)

        # Convert class labels to 0 and 1
        drebin_df['class'] = drebin_df['class'].map({'B': 0, 'S': 1})

        # Remove rows with non-numeric values
        drebin_df = drebin_df.apply(pd.to_numeric, errors='coerce')
        drebin_df = drebin_df.dropna()

        self.labels = drebin_df['class']
        self.drebin_df = drebin_df

        # Load permission and API names
        with open(config.permission_file, 'r') as file:
            self.permissions_features = [line.strip() for line in file.readlines()]

        with open(config.api_file, 'r') as file:
            self.api_features = [line.strip() for line in file.readlines()]

        # Combine all feature names (permissions + API calls)
        self.all_features = self.permissions_features + self.api_features

    def get_permissions_features(self) -> pd.DataFrame:
        """
        Returns the features of the Drebin dataset related to manifest permissions.
        """
        return self.drebin_df[self.permissions_features]

    def get_all_features(self) -> pd.DataFrame:
        """
        Returns the features of the Drebin dataset, including both manifest permissions
        and API call signature features.
        """
        return self.drebin_df[self.all_features]

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the dataset into training and testing sets for both Manifest Permissions
        and combined features (Manifest Permissions + API Calls).
        """
        # Get the features for manifest permissions and all features
        X_permissions = self.get_permissions_features()
        X_all_features = self.get_all_features()

        # Split data into training and test sets for manifest permissions only
        X_train_perm, X_test_perm, y_train, y_test = train_test_split(
            X_permissions, self.labels, test_size=config.test_size, random_state=42
        )

        # Split data into training and test sets for all features (permissions + API calls)
        X_train_all, X_test_all, _, _ = train_test_split(
            X_all_features, self.labels, test_size=config.test_size, random_state=42
        )

        return X_train_perm, X_test_perm, X_train_all, X_test_all, y_train, y_test
