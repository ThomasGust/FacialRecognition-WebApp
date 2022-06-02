from multiprocessing.dummy import Value
from typing import List, Dict, Tuple, Type
import os
import json
import uuid
import numpy as np
import dlib
import cv2
import shutil
import string
import random

def draw_bounding_box(image, bbox: List[int], color: Tuple = (0, 255, 0)):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image

def draw_annotation(image, name: str, bbox: List[int], color: Tuple = (0, 255, 0)):
    draw_bounding_box(image, bbox, color=color)
    x1, _, x2, y2 = bbox
    cv2.rectangle(image, (x1, y2 - 20), (x2, y2), color, cv2.FILLED)
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(image, name, (x1 + 6, y2 - 6), font, 0.6, (0, 0, 0), 2)

class Cache:
    def __init__(self, data: List[Dict] = None):
        if data is not None and (type(data) is not list or (len(data) and type(data[0]) is not dict)):
            raise TypeError()

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
            raise TypeError("DATA IS NOT A DICT")
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


class JStorage:

    def __init__(self, db_loc: str = "./data/facial_data_db.json"):
        self.db_loc = db_loc

    def add_data(self, face_data: Dict):
        data = []
        base_path, _ = os.path.split(self.db_loc)

        if not os.path.exists(base_path):
            os.makedirs(base_path)
        if os.path.exists(self.db_loc):
            data = self.get_all_data()
        try:
            data.append(face_data)
            self.save_data(data=data)
        except Exception as exc:
            raise exc

    def get_all_data(self) -> List:
        if not os.path.exists(self.db_loc):
            raise FileNotFoundError("Could not find the database location")
        try:
            with open(self.db_loc, "r") as f:
                data = json.load(f)
                return self.sanitize_data(data)
        except Exception as exc:
            return json.loads("[]")

    def save_data(self, data: Dict = None):
        if data is not None:
            with open(self.db_loc, "w") as f:
                json.dump(data, f)

    def sanitize_data(self, data: List[Dict]) -> List[Dict]:
        for face_data in data:
            face_data["encoding"] = tuple(face_data["encoding"])
        return data


class FaceData:
    def __init__(self, persistent_data_loc="data/facial_data.json") -> None:
        self.db_handler = JStorage(persistent_data_loc)
        saved_data = []
        try:
            saved_data = self.db_handler.get_all_data()
        except Exception as e:
            raise(e)
        try:
            self.cache_handler = Cache(saved_data)
        except Exception as e:
            raise e

    def add_facial_data(self, facial_data):
        self.cache_handler.add_data(face_data=facial_data)
        self.db_handler.add_data(face_data=facial_data)

    def get_all_facial_data(self):
        return self.cache_handler.get_all_data()

class FaceDetectorDlib:
    cnn_model_filename = "mmod_human_face_detector.dat"

    def __init__(self, model_loc: str = "models", model_type: str = "hog"):
        try:
            if model_type == "hog":
                self.face_detector = dlib.get_frontal_face_detector()
            else:
                cnn_model_path = os.path.join(
                    model_loc, FaceDetectorDlib.cnn_model_filename
                )
                if not os.path.exists(cnn_model_path):
                    raise FileNotFoundError()
                self.face_detector = dlib.cnn_face_detection_model_v1(cnn_model_path)
            self.model_type = model_type
        except Exception as e:
            raise e

    def detect_faces(self, image, num_upscaling: int = 1) -> List[List[int]]:
        if not image is None or not (len(image.shape) != 3 or image.shape[-1] != 3):
            raise Exception()
        return [
            self.dlib_rectangle_to_list(bbox)
            for bbox in self.face_detector(image, num_upscaling)
        ]

    def dlib_rectangle_to_list(self, dlib_bbox) -> List[int]:
        if type(dlib_bbox) == dlib.mmod_rectangle:
            dlib_bbox = dlib_bbox.rect
        x1, y1 = dlib_bbox.tl_corner().x, dlib_bbox.tl_corner().y
        width, height = dlib_bbox.width(), dlib_bbox.height()
        x2, y2 = x1 + width, y1 + height

        return [x1, y1, x2, y2]


class FaceRecognition:
    keypoints_model_path = "shape_predictor_5_face_landmarks.dat"
    face_recog_model_path = "dlib_face_recognition_resnet_model_v1.dat"

    def __init__(self, json_data, model_loc: str = "./models", persistent_data_loc="data/facial_data.json", face_detection_threshold: int = 0.8,) -> None:
        keypoints_model_path = os.path.join(
            model_loc, FaceRecognition.keypoints_model_path
        )
        face_recog_model_path = os.path.join(
            model_loc, FaceRecognition.face_recog_model_path
        )
        if not (os.path.exists(keypoints_model_path) or os.path.exists(face_recog_model_path)):
            raise FileNotFoundError()
        self.face_detector = FaceDetectorDlib()
        self.face_detection_threshold = face_detection_threshold

        self.keypoints_detector = dlib.shape_predictor(keypoints_model_path)
        self.face_recognizor = dlib.face_recognition_model_v1(face_recog_model_path)
        self.datastore = FaceData(persistent_data_loc=persistent_data_loc)


    def register_face(self, image=None, name: str = None, bbox: List[int] = None):
        if not image is None or not (len(image.shape) != 3 or image.shape[-1] != 3) or name is None:
            raise TypeError

        image = image.copy()
        face_encoding = None

        try:
            if bbox is None:
                bboxes = self.face_detector.detect_faces(image=image)
                if len(bboxes) == 0:
                    raise ValueError()
                bbox = bboxes[0]
            face_encoding = self.get_facial_fingerprint(image, bbox)

            facial_data = {
                "id": str(uuid.uuid4()),
                "encoding": tuple(face_encoding.tolist()),
                "name": name,
            }
            self.save_facial_data(facial_data)
        except Exception as exc:
            raise exc
        return facial_data


    def save_facial_data(self, facial_data: Dict = None) -> bool:
        if facial_data is not None:
            self.datastore.add_facial_data(facial_data=facial_data)
            return True
        return False

    def get_registered_faces(self) -> List[Dict]:
        return self.datastore.get_all_facial_data()

    def recognize_faces(self, image, threshold: float = 0.6, bboxes: List[List[int]] = None):
         if image is None:
            return TypeError()
         image = image.copy()
         if bboxes is None:
             bboxes = self.face_detector.detect_faces(image=image)
         if len(bboxes) == 0:
             raise ValueError()
         all_facial_data = self.datastore.get_all_facial_data()
         matches = []
         for bbox in bboxes:
            face_encoding = self.get_facial_fingerprint(image, bbox)
            match, min_dist = None, 10000000
            for face_data in all_facial_data:
                dist = self.euclidean_distance(face_encoding, face_data["encoding"])
                if dist <= threshold and dist < min_dist:
                    match = face_data
                    min_dist = dist
            matches.append((bbox, match, min_dist))
         return matches
    def get_facial_fingerprint(self, image, bbox: List[int] = None) -> List[float]:
        if bbox is None:
            raise ValueError
        bbox = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
        face_keypoints = self.keypoints_detector(image, bbox)
        face_encoding = self.get_face_encoding(image, face_keypoints)
        return face_encoding

    def get_face_encoding(self, image, face_keypoints: List):
        encoding = self.face_recognizor.compute_face_descriptor(
            image, face_keypoints, 1
        )
        return np.array(encoding)

    def euclidean_distance(self, vector1: Tuple, vector2: Tuple):
        return np.linalg.norm(np.array(vector1) - np.array(vector2))


class WebFaceRegistry:
    def __init__(
            self, model_loc: str = "models", persistent_db_path: str = "data/facial_data.json", face_detection_threshold: float = 0.8) -> None:
        self.model_loc = model_loc
        self.face_detection_threshold = face_detection_threshold

        self.face_recognizer = FaceRecognition(model_loc=model_loc, persistent_data_loc=persistent_db_path, face_detection_threshold=face_detection_threshold, face_detector="dlib")
        self.face_detector = FaceDetectorDlib()

    def batch_register_face_web(self, faces, key):
        jdirpath = f"face_recognition\\tmp\json\{key}"
        jdirpath = os.path.join(os.getcwd(), jdirpath)
        if not os.path.isdir(jdirpath):
            os.mkdir(jdirpath)
        dirpath = f"tmp\images\{key}"
        paths = [f'{os.path.join(dirpath, n)}.jpg' for _, (__, n) in enumerate(faces)]
        j_path = f'{jdirpath}\\facial_data.json'
        fp = open(j_path, mode="x+")
        fp.close()
        self.face_recognizer = FaceRecognition(persistent_data_loc=j_path)
        for i in range(paths):
            f, n = faces[i]
            self.face_recognizer.register_face(
                image=cv2.cvtColor(f, cv2.COLOR_BGR2RGB), name=n
            )
        try:
            with open(j_path, "rb") as f:
                j = json.loads(f.read())
        except Exception as e:
            print(e)
            raise e 
        shutil.rmtree(dirpath, ignore_errors=True)
        shutil.rmtree(jdirpath, ignore_errors=True)
        return j

    def get_rand_string(self, length):
        str = string.ascii_lowercase
        return ''.join(random.choice(str) for _ in range(length))


class OnWebFaceRecognition:
    keypoints_model_path = "shape_predictor_5_face_landmarks.dat"
    face_recog_model_path = "dlib_face_recognition_resnet_model_v1.dat"

    def __init__(
            self,
            json_data,
            model_loc: str = "./models",
            face_detection_threshold: int = 0.99,
            face_detector: str = "dlib",
    ) -> None:
        keypoints_model_path = os.path.join(
            model_loc, FaceRecognition.keypoints_model_path
        )
        face_recog_model_path = os.path.join(
            model_loc, FaceRecognition.face_recog_model_path
        )
        if not (
                os.path.exists(keypoints_model_path) or os.path.exists(face_recog_model_path)
        ):
            raise FileNotFoundError
        self.face_detector = FaceDetectorDlib()
        self.face_detection_threshold = face_detection_threshold

        self.keypoints_detector = dlib.shape_predictor(keypoints_model_path)
        self.face_recognizor = dlib.face_recognition_model_v1(face_recog_model_path)
        self.json_data = json_data
        #self.datastore = FaceDataStore(persistent_data_loc=persistent_data_loc)

    def register_face(self, image=None, name: str = None, bbox: List[int] = None):
        if not image is None or not (len(image.shape) != 3 or image.shape[-1] != 3) or name is None:
            raise TypeError

        image = image.copy()
        face_encoding = None

        try:
            if bbox is None:
                bboxes = self.face_detector.detect_faces(image=image)
                if len(bboxes) == 0:
                    raise ValueError()
                bbox = bboxes[0]
            face_encoding = self.get_facial_fingerprint(image, bbox)

            facial_data = {
                "id": str(uuid.uuid4()),
                "encoding": tuple(face_encoding.tolist()),
                "name": name,
            }
            self.save_facial_data(facial_data)
        except Exception as exc:
            raise exc
        return facial_data

    def save_facial_data(self, facial_data: Dict = None) -> bool:
        if facial_data is not None:
            self.datastore.add_facial_data(facial_data=facial_data)
            return True
        return False

    def get_registered_faces(self) -> List[Dict]:
        return self.datastore.get_all_facial_data()

    def recognize_faces(
            self, image,threshold: float =  0.6, bboxes: List[List[int]] = None
    ):
        if image is None:
            return TypeError()
        image = image.copy()

        if bboxes is None:
            bboxes = self.face_detector.detect_faces(image=image)
        if len(bboxes) == 0:
            raise ValueError()
        matches = []
        for bbox in bboxes:
            face_encoding = self.get_facial_fingerprint(image, bbox)
            match, min_dist = None, 10000000

            for face_data in self.json_data:
                dist = self.euclidean_distance(face_encoding, face_data["encoding"])
                if dist <= threshold and dist < min_dist:
                    match = face_data
                    min_dist = dist
            matches.append((bbox, match, min_dist))
        return matches

    def get_facial_fingerprint(self, image, bbox: List[int] = None) -> List[float]:
        if bbox is None:
            raise ValueError()
        bbox = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
        face_keypoints = self.keypoints_detector(image, bbox)
        face_encoding = self.get_face_encoding(image, face_keypoints)
        return face_encoding

    def get_face_encoding(self, image, face_keypoints: List):
        encoding = self.face_recognizor.compute_face_descriptor(
            image, face_keypoints, 1
        )
        return np.array(encoding)

    def euclidean_distance(self, vector1: Tuple, vector2: Tuple):
        return np.linalg.norm(np.array(vector1) - np.array(vector2))