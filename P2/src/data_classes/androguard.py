from androguard.misc import AnalyzeAPK


class AndroguardProcessor:
    """
    Class to process APK files using Androguard, extract permissions and API calls.
    """

    def __init__(self, apk_file: str):
        """
        Initializes the AndroguardProcessor with the given APK file.

        :param apk_file: Path to the APK file to be analyzed.
        :return:
        """
        self.apk_file = apk_file
        self.apk = None
        self.detailed_analysis = None

    def load_apk(self) -> None:
        """
        Loads and analyzes the APK file using AnalyzeAPK.
        :return:
        """
        try:
            self.apk, _, self.detailed_analysis = AnalyzeAPK(self.apk_file)
        except Exception as e:
            raise ValueError(f"Error loading APK: {e}")

    def get_permissions(self) -> list:
        """
        Extracts and returns the permissions from the APK file.

        :return: List of permissions as strings.
        """
        # Get the permissions from the APK
        permision = self.apk.get_permissions()

        # Extract the permission name from the full permission string
        return [p.split('.')[-1] for p in permision]

    def get_api_calls(self) -> list:
        """
        Extracts API calls from the APK file using Androguard's detailed analysis.

        :return: List of API calls as strings.
        """
        api_calls = []

        for class_analysis in self.detailed_analysis.get_external_classes():
            class_name = class_analysis.name.replace(';', '')
            class_methods = class_analysis.get_methods()

            # For each method in the class, construct the full method name
            for method in class_methods:
                api_call = f"{class_name.replace('/', '.')}.{method.name}"
                api_calls.append(api_call)

        return api_calls

    def analyze(self) -> None:
        """
        Analyze the APK and print basic information like package name and permissions.

        :return:
        """
        print(f"Package Name: {self.apk.get_package()}")
        permissions = self.get_permissions()
        print(f"Permissions: {permissions}")
        api_calls = self.get_api_calls()
        print(f"API Calls: {api_calls[:10]}...")