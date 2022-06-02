
class ModelFileMissing(Exception):
    def __init__(self):
        self.message = "Model file missing!!"


class NoFaceDetected(Exception):
    def __init__(self) -> None:
        self.message = "No face found in image"


class MultipleFacesDetected(Exception):
    def __init__(self) -> None:
        self.message = "Multiple faces found in image"


class InvalidImage(Exception):
    def __init__(self) -> None:
        self.message = "Invalid Image"


class DatabaseFileNotFound(Exception):
    def __init__(self) -> None:
        self.message = "Database file not found"


class InvalidCacheInitializationData(Exception):
    def __init__(self) -> None:
        self.message = "Invalid data structure. Please suppply a list"


class NotADictionary(Exception):
    def __init__(self) -> None:
        self.message = "Invalid data structure. Please suppply a dict"


class NoNameProvided(Exception):
    def __init__(self) -> None:
        self.message = "Please provide a name for registering face"


class PathNotFound(Exception):
    def __init__(self) -> None:
        self.message = "Path couldn't be found. Please check"


class FaceMissing(Exception):
    def __init__(self) -> None:
        self.message = "Face not found"