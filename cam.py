import threading
import binascii
from time import sleep
from utils import base64_to_pil_image, pil_image_to_base64
import cv2
from face_recognition import OnWebFaceRecognition, draw_annotation
import numpy as np
import base64

class Annotator:

    def __init__(self, mjson=None, mdt=None):
        self.mjson = mjson
        self.mdt = mdt
    
    def apply_filter(self, frame):
        if self.mjson != None and self.mdt != None:
            fr = OnWebFaceRecognition(json_data=self.mjson, face_detection_threshold=self.mdt)
            smaller_frame = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB)
            try:
                matches = fr.recognize_faces(
                    image=smaller_frame, threshold=0.6, bboxes=None
                )
                for face_bbox, match, _ in matches:
                    name = match["name"] if match is not None else "Unknown"
                    draw_annotation(frame, name, int(1 / 0.5) * np.array(face_bbox))

            except Exception as e:
                raise e
            _, buffer = cv2.imencode(".jpg", frame)
            return base64.b64encode(buffer)
        return frame

class Camera(object):
    def __init__(self, f=Annotator()):
        self.to_process = []
        self.to_output = []
        self.filter = f

        thread = threading.Thread(target=self.keep_processing, args=())
        thread.daemon = True
        thread.start()

    def process_one(self):
        if not self.to_process:
            return
        input_str = self.to_process.pop(0)
        input_img = base64_to_pil_image(input_str)
        output_str = self.filter.apply_filter(input_img)
        self.to_output.append(binascii.a2b_base64(output_str))

    def keep_processing(self):
        while True:
            self.process_one()
            sleep(0.01)
    def enqueue_input(self, input):
        self.to_process.append(input)

    def get_frame(self):
        while not self.to_output:
            sleep(0.05)
        return self.to_output.pop(0)