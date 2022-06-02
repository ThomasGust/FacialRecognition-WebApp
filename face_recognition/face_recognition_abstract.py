from abc import ABC, abstractmethod


class FaceDetector(ABC):
    @abstractmethod
    def detect_faces(self):
        pass


class PersistentStorage(ABC):
    @abstractmethod
    def add_data(self):
        pass

    @abstractmethod
    def get_all_data(self):
        pass


class InMemoryCache(ABC):
    @abstractmethod
    def add_data(self):
        pass

    @abstractmethod
    def get_all_data(self):
        pass
