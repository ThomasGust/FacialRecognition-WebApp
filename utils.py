import cv2
from face_recognition import OnWebFaceRecognition, draw_annotation
import numpy as np

def generate_frames(mjson, mdt):
    camera = cv2.VideoCapture(0)
    fr = OnWebFaceRecognition(json_data=mjson, face_detection_threshold=mdt)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            smaller_frame = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=0.5, fy=0.5), cv2.COLOR_BGR2RGB)
            try:
                matches = fr.recognize_faces(
                    image=smaller_frame, threshold=0.6, bboxes=None
                )
                for face_bbox, match, _ in matches:
                    name = match["name"] if match is not None else "Unknown"
                    draw_annotation(frame, name, int(1 / 0.5) * np.array(face_bbox))

            except Exception:
                pass
            _, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def allowed_extension(filename, allowed_extensions):
    if not "." in filename:
        return False
    else:
        ext = filename.rsplit(".", 1)[1]
        if ext.upper() in allowed_extensions:
            return True
        else:
            return False

def get_f_name(filename):
    if not "." in filename:
        raise Exception
    else:
        name = filename.split(".")[0]
    return name