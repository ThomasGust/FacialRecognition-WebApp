from face_recognition.face_recognition_exceptions import InvalidCacheInitializationData, DatabaseFileNotFound, NotADictionary
from face_recognition.face_recognition_logger import LoggerFactory
from face_recognition.face_recognition_utils import path_exists
from face_recognition.face_recognition_abstract import PersistentStorage, InMemoryCache
import sys
import os
from typing import Dict, List, Tuple
import json

try:
    logger_ob = LoggerFactory(logger_name=__name__)
    logger = logger_ob.get_logger()
    logger.info("{} loaded...".format(__name__))
    sys.excepthook = logger_ob.uncaught_exception_hook
except Exception as exc:
    raise exc


class SimpleCache(InMemoryCache):
    def __init__(self, data: List[Dict] = None):
        if data is not None and (
            type(data) is not list or (len(data) and type(data[0]) is not dict)
        ):
            raise InvalidCacheInitializationData

        self.facial_data = set()
        if data:
            for face_data in data:
                self.facial_data.add(self.serialize_dict(face_data))

    def add_data(self, face_data: Dict):
        facial_data = self.serialize_dict(face_data)
        self.facial_data.add(facial_data)

    def get_all_data(self) -> List[Dict]:
        return self.deserialize_data(self.facial_data)

    def delete_data(self, face_id: str):
        for data in self.facial_data:
            for key_pair in data:
                if face_id in key_pair:
                    self.facial_data.discard(data)
                    return True
        return False

    def serialize_dict(self, data: Dict) -> Tuple[Tuple]:
        if "encoding" in data and type(data["encoding"]) is list:
            data["encoding"] = tuple(data["encoding"])
        if type(data) is not dict:
            raise NotADictionary
        return tuple(sorted(data.items()))

    def deserialize_data(self, data) -> List[Dict]:
        facial_data = []
        for face_data in data:
            facial_data.append({item[0]: item[1] for item in face_data})

        return facial_data

    def get_details(self) -> List[Dict]:
        facial_data = self.get_all_data()
        facial_data = [
            (face_data["id"], face_data["name"]) for face_data in facial_data
        ]
        return facial_data


class JSONStorage(PersistentStorage):

    def __init__(self, db_loc: str = "./data/facial_data_db.json"):
        self.db_loc = db_loc

    def add_data(self, face_data: Dict):
        data = []
        base_path, filename = os.path.split(self.db_loc)

        if not path_exists(base_path):
            logger.info("DB path doesn't exist! Attempting path creation...")
            os.makedirs(base_path)
        if os.path.exists(self.db_loc):
            data = self.get_all_data()
        try:
            data.append(face_data)
            self.save_data(data=data)
            logger.info("Data saved to DB...")
        except Exception as exc:
            raise exc

    def get_all_data(self) -> List:
        if not path_exists(self.db_loc):
            raise DatabaseFileNotFound
        try:
            with open(self.db_loc, "r") as f:
                data = json.load(f)
                return self.sanitize_data(data)
        except Exception as exc:
            return json.loads("[]")

    def delete_data(self, face_id: str) -> bool:
        all_data = self.get_all_data()
        num_entries = len(all_data)
        for idx, face_data in enumerate(all_data):
            for key_pair in face_data.items():
                if face_id in key_pair:
                    all_data.pop(idx)

        if num_entries != len(all_data):
            self.save_data(data=all_data)
            logger.info(
                ("{} face(s) deleted and updated" " data saved to DB...").format(
                    num_entries - len(all_data)
                )
            )
            return True
        return False

    def save_data(self, data: Dict = None):
        if data is not None:
            with open(self.db_loc, "w") as f:
                json.dump(data, f)

    def sanitize_data(self, data: List[Dict]) -> List[Dict]:
        for face_data in data:
            face_data["encoding"] = tuple(face_data["encoding"])
        return data


class FaceDataStore:
    def __init__(self, persistent_data_loc="data/facial_data.json") -> None:
        self.db_handler = JSONStorage(persistent_data_loc)
        saved_data = []
        try:
            saved_data = self.db_handler.get_all_data()
            logger.info(
                "Data loaded from DB with {} entries...".format(len(saved_data))
            )
        except Exception as e:
            logger.info("No existing DB file found!!")
        try:
            self.cache_handler = SimpleCache(saved_data)
        except InvalidCacheInitializationData:
            raise InvalidCacheInitializationData

    def add_facial_data(self, facial_data):
        self.cache_handler.add_data(face_data=facial_data)
        self.db_handler.add_data(face_data=facial_data)

    def remove_facial_data(self, face_id: str = None):
        self.cache_handler.delete_data(face_id=face_id)
        self.db_handler.delete_data(face_id=face_id)

    def get_all_facial_data(self):
        return self.cache_handler.get_all_data()