import os
import sys
from typing import List, Dict, Tuple
import uuid

import dlib
import numpy as np
import time
import traceback

from face_recognition.face_recognition_exceptions import PathNotFound, ModelFileMissing, NoNameProvided, FaceMissing, \
    NoFaceDetected
from face_recognition.face_recognition_abstract import FaceDetector
from face_recognition.face_recognition_logger import LoggerFactory
from face_recognition.face_recognition_utils import *
from face_recognition.face_recognition_datastore import FaceDataStore
import string
import random
import json
import time
import shutil

logger = None
try:
    logger_ob = LoggerFactory(logger_name=__name__)
    logger = logger_ob.get_logger()
    logger.info("{} loaded...".format(__name__))
    sys.excepthook = logger_ob.uncaught_exception_hook
except Exception as exc:
    raise exc


class FaceDetectorDlib(FaceDetector):
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
                    raise ModelFileMissing
                self.face_detector = dlib.cnn_face_detection_model_v1(cnn_model_path)
            self.model_type = model_type
            logger.info("dlib: {} face detector loaded...".format(self.model_type))
        except Exception as e:
            raise e

    def detect_faces(self, image, num_upscaling: int = 1) -> List[List[int]]:
        if not is_valid_img(image):
            raise InvalidImage
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

    def __init__(
            self,
            model_loc: str = "./models",
            persistent_data_loc="data/facial_data.json",
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
                path_exists(keypoints_model_path) or path_exists(face_recog_model_path)
        ):
            raise ModelFileMissing
        self.face_detector = FaceDetectorDlib()
        self.face_detection_threshold = face_detection_threshold

        self.keypoints_detector = dlib.shape_predictor(keypoints_model_path)
        self.face_recognizor = dlib.face_recognition_model_v1(face_recog_model_path)
        self.datastore = FaceDataStore(persistent_data_loc=persistent_data_loc)

    def register_face(self, image=None, name: str = None, bbox: List[int] = None):
        if not is_valid_img(image) or name is None:
            raise NoNameProvided if name is None else InvalidImage

        image = image.copy()
        face_encoding = None

        try:
            if bbox is None:
                bboxes = self.face_detector.detect_faces(image=image)
                if len(bboxes) == 0:
                    raise NoFaceDetected
                bbox = bboxes[0]
            face_encoding = self.get_facial_fingerprint(image, bbox)

            facial_data = {
                "id": str(uuid.uuid4()),
                "encoding": tuple(face_encoding.tolist()),
                "name": name,
            }
            self.save_facial_data(facial_data)
            logger.info("Face registered with name: {}".format(name))
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
            self, image, threshold: float = 0.6, bboxes: List[List[int]] = None
    ):
        if image is None:
            return InvalidImage
        image = image.copy()

        if bboxes is None:
            bboxes = self.face_detector.detect_faces(image=image)
        if len(bboxes) == 0:
            return image
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
            raise FaceMissing
        bbox = convert_to_dlib_rectangle(bbox)
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
                path_exists(keypoints_model_path) or path_exists(face_recog_model_path)
        ):
            raise ModelFileMissing
        self.face_detector = FaceDetectorDlib()
        self.face_detection_threshold = face_detection_threshold

        self.keypoints_detector = dlib.shape_predictor(keypoints_model_path)
        self.face_recognizor = dlib.face_recognition_model_v1(face_recog_model_path)
        self.json_data = json_data
        #self.datastore = FaceDataStore(persistent_data_loc=persistent_data_loc)

    def register_face(self, image=None, name: str = None, bbox: List[int] = None):
        if not is_valid_img(image) or name is None:
            raise NoNameProvided if name is None else InvalidImage

        image = image.copy()
        face_encoding = None

        try:
            if bbox is None:
                bboxes = self.face_detector.detect_faces(image=image)
                if len(bboxes) == 0:
                    raise NoFaceDetected
                bbox = bboxes[0]
            face_encoding = self.get_facial_fingerprint(image, bbox)

            facial_data = {
                "id": str(uuid.uuid4()),
                "encoding": tuple(face_encoding.tolist()),
                "name": name,
            }
            self.save_facial_data(facial_data)
            logger.info("Face registered with name: {}".format(name))
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
            return InvalidImage
        image = image.copy()

        if bboxes is None:
            bboxes = self.face_detector.detect_faces(image=image)
        if len(bboxes) == 0:
            return None
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
            raise FaceMissing
        bbox = convert_to_dlib_rectangle(bbox)
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


class FaceRecognitionVideo:
    def __init__(
            self,
            model_loc: str = "models",
            persistent_db_path: str = "data/facial_data.json",
            face_detection_threshold: float = 0.8,
    ) -> None:
        self.model_loc = model_loc
        self.face_detection_threshold = face_detection_threshold

        self.face_recognizer = FaceRecognition(
            model_loc=model_loc,
            persistent_data_loc=persistent_db_path,
            face_detection_threshold=face_detection_threshold,
            face_detector="dlib",
        )
        self.face_detector = FaceDetectorDlib()

    def recognize_face_video(
            self,
            video_path: str = None,
            detection_interval: int = 15,
            save_output: bool = False,
            preview: bool = False,
            output_path: str = "data/output.mp4",
            resize_scale: float = 0.5,
            verbose: bool = True,
    ) -> None:

        if video_path is None:
            video_path = 0
        elif not path_exists(video_path):
            raise FileNotFoundError

        cap, video_writer = None, None

        try:
            cap = cv2.VideoCapture(video_path)
            video_writer = get_video_writer(cap, output_path)
            frame_num = 1
            matches, name, match_dist = [], None, None

            t1 = time.time()
            logger.info("Enter q to exit...")

            while True:
                status, frame = cap.read()
                if not status:
                    break
                try:
                    if video_path == 0:
                        frame = cv2.flip(frame, 2)

                    if frame_num % detection_interval == 0:
                        smaller_frame = convert_to_rgb(
                            cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                        )
                        matches = self.face_recognizer.recognize_faces(
                            image=smaller_frame, threshold=0.6, bboxes=None
                        )
                    if verbose:
                        self.annotate_facial_data(matches, frame, resize_scale)
                    if save_output:
                        video_writer.write(frame)
                    if preview:
                        cv2.imshow("Preview", cv2.resize(frame, (680, 480)))

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                except Exception:
                    pass
                frame_num += 1

            t2 = time.time()
            logger.info("Time:{}".format((t2 - t1) / 60))
            logger.info("Total frames: {}".format(frame_num))
            logger.info("Time per frame: {}".format((t2 - t1) / frame_num))

        except Exception as exc:
            raise exc
        finally:
            cv2.destroyAllWindows()
            cap.release()
            video_writer.release()

    def register_face_webcam(
            self, name: str = None, detection_interval: int = 5
    ) -> bool:
        if name is None:
            raise NoNameProvided

        cap = None
        try:
            cap = cv2.VideoCapture(0)
            frame_num = 0

            while True:
                status, frame = cap.read()
                if not status:
                    break

                if frame_num % detection_interval == 0:
                    bboxes = self.face_detector.detect_faces(image=frame)
                    try:
                        if len(bboxes) == 1:
                            facial_data = self.face_recognizer.register_face(
                                image=frame, name=name, bbox=bboxes[0]
                            )
                            if facial_data:
                                draw_bounding_box(frame, bboxes[0])
                                cv2.imshow("Registered Face", frame)
                                cv2.waitKey(0)
                                logger.info("Press any key to continue......")
                                break
                    except Exception as exc:
                        traceback.print_exc(file=sys.stdout)
                frame_num += 1
        except Exception as exc:
            raise exc
        finally:
            cv2.destroyAllWindows()
            cap.release()

    def register_face_path(self, img_path: str, name: str) -> None:
        if not path_exists(img_path):
            raise PathNotFound
        try:
            img = cv2.imread(img_path)
            facial_data = self.face_recognizer.register_face(
                image=convert_to_rgb(img), name=name
            )
            if facial_data:
                logger.info("Face registered...")
                return True
            return False
        except Exception as exc:
            raise exc

    def batch_register_face_paths(self, dirpath) -> None:
        if not path_exists(dirpath):
            raise PathNotFound
        try:
            fs = os.listdir(dirpath)

            for f in fs:
                p = os.path.join(dirpath, f)
                n = f.removesuffix(".jpg")
                self.register_face_path(img_path=p, name=n)

        except Exception as exc:
            raise exc
    
    def u_batch_register_face_paths(self, dirpath, recog) -> None:
        self.face_recognizer = recog
        if not path_exists(dirpath):
            os.mkdir(dirpath)
        try:
            fs = os.listdir(dirpath)

            for f in fs:
                p = os.path.join(dirpath, f)
                n = f.removesuffix(".jpg")
                self.register_face_path(img_path=p, name=n)

        except Exception as exc:
            raise exc
        
    def batch_register_face_web(self, faces, key):
        jdirpath = f"face_recognition\\tmp\json\{key}"
        #fp = open("face_recognition\\tmp\json\\testdir\okasdfihbasdiybflashd.txt", "x+")
        jdirpath = os.path.join(os.getcwd(), jdirpath)
        if not os.path.isdir(jdirpath):
            os.mkdir(jdirpath)
        
        dirpath = f"tmp\images\{key}"
        paths = [f'{os.path.join(dirpath, n)}.jpg' for _, (__, n) in enumerate(faces)]
        
        j_path = f'{jdirpath}\\facial_data.json'

        #print(os.path.isdir(jdirpath))
        fp = open(j_path, mode="x+")
        fp.close()

        self.face_recognizer = FaceRecognition(persistent_data_loc=j_path)


        for i, p in enumerate(paths):
            f, n = faces[i]
            #cv2.imwrite(p, f)
            facial_data = self.face_recognizer.register_face(
                image=convert_to_rgb(f), name=n
            )


        #self.u_batch_register_face_paths(dirpath=dirpath, recog=FaceRecognition(persistent_data_loc=j_path))
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
        """Generate a random string"""
        str = string.ascii_lowercase
        return ''.join(random.choice(str) for _ in range(length))

    def annotate_facial_data(
            self, matches: List[Dict], image, resize_scale: float
    ) -> None:
        for face_bbox, match, dist in matches:
            name = match["name"] if match is not None else "Unknown"
            draw_annotation(image, name, int(1 / resize_scale) * np.array(face_bbox))

    def regenerate_registry(self, path_of_dir, f_data, o):
        if os.path.isfile("data/facial_data.json"):
            os.remove("data/facial_data.json")

        if os.path.isfile("data/output.mp4"):
            os.remove("data/output.mp4")

        self.batch_register_face_paths(path_of_dir)


