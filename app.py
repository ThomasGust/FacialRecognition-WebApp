import cv2
import os
from flask import Flask, render_template, Response, request, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import numpy as np
from face_recognition import FRV
import hashlib
from config import ProductionConfig
from utils import *
from models import get_models
import base64
import io
from flask_socketio import SocketIO, emit
from cam import Annotator, Camera
from imageio import imread
import socket
import threading
from utils import generate_frames
import sys
from pyngrok import ngrok

app = Flask(__name__, template_folder="templates")
app.config.from_object(ProductionConfig())
db = SQLAlchemy(app)
socketio = SocketIO(app)
jmods, users = get_models(db=db)


@socketio.on('input-image')
def test_message(input):
    
    #input = input.split(",")[1]
    with app.app_context():
        umodi = session['umodi']
        model = jmods.query.get(umodi)
        mjson = model.model
        mdt = model.mdt
    print("SUCCESFULLY LOADED CONTEXT")
    
    camera = Camera(Annotator(mjson=mjson, mdt=mdt))
    print("LOADED CAMERA")
    input = input.split(",")[1]
    print("SPLIT INPUT")
    image_data = camera.filter.apply_filter(base64_to_cv2_image(input))
    img = base64.b64decode(image_data)
    buf_arr = np.fromstring(img, dtype=np.uint8)
    img = cv2.imdecode(buf_arr, cv2.IMREAD_UNCHANGED)
    cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, buffer = cv2.imencode('.jpg', cv2_img)
    b = base64.b64encode(buffer)
    b = b.decode()
    image_data = "data:image/jpeg;base64," + b

    print("OUTPUT " + image_data)
    emit('out-image-event', {'image_data': image_data}, namespace='/v-stream')
    print("IMAGE RECIEVED")
    emit('out-image-event', {'image_data': input})



@app.route("/")
@app.route("/home")
def index():
    try:
        if session['username']:
            return render_template("index.html")
        else:
            return(redirect(url_for("sign_in")))
    except Exception as e:
        print(e)
        return(redirect(url_for("sign_in")))


@app.route("/video")
def video():
    jm = jmods.query.get(session['umodi'])
    return Response(generate_frames(mjson=jm.model, mdt=jm.mdt), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/model-dashboard")
def model_dashboard():
    if session['username']:
        models = jmods.query.filter_by(created_by=session['username']).all()
        return render_template("model_dashboard.html", models=models)
    else:
        return(redirect(url_for("sign_in")))

@app.route("/add-model", methods=["GET", "POST"])
def add_model():
    if request.method == "POST":
        if request.files:
            d = request.form.to_dict()
            files = request.files.getlist("images")
            print(files)
            imgs = []

            for file in files:
                if file.filename == "":
                    print("FILE MUST HAVE A FILENAME!!!")
                    return redirect(url_for("model_dashboard"))
                
                elif not allowed_extension(file.filename, app.config['ALLOWED_IMAGE_EXTENSIONS']):
                    print("THAT EXTENSION IS NOT ALLOWED")
                    return redirect(url_for("model_dashboard"))
                
                else:
                    npimg = np.fromstring(file.read(), np.uint8)
                    pimg = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                    imgs.append((pimg, get_f_name(file.filename)))
            
            mname = d['mname']
            mdt = d['mdt']
            fr = FRV()
            jsondata = fr.batch_register_face_web(imgs, key=fr.get_rand_string(length=100))

            
            jm = jmods(session['username'], mname, jsondata, mdt)
            db.session.add(jm)
            db.session.commit()
            
    return redirect(url_for("model_dashboard"))

@app.route("/model-inference-live", methods=["GET", "POST"])
def model_inference():
    if session['username']:
        return render_template("model_inference.html")
    else:
        return(redirect(url_for("sign_in")))

@app.route("/model-inference", methods=["GET", "POST"])
def mif():
    if session['username']:
        models = jmods.query.filter_by(created_by=session['username']).all()
        return render_template("model_inference_form.html", models=models)
    else:
        return(redirect(url_for("sign_in")))

@app.route("/code-explanation")
def code_explanation():
    if session['username']:
        return render_template("explanation.html")
    else:
        return(redirect(url_for("sign_in")))


@app.route("/reflection")
def reflection():
    if session['username']:
        return render_template("reflection.html")
    else:
        return(redirect(url_for("sign_in")))


@app.route("/about-project")
def about():
    if session['username']:
        return render_template("about.html")
    else:
        return(redirect(url_for("sign_in")))

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":
        #session.permanent = True
        d = request.form.to_dict()

        u = d['uname']
        p = d['pwd']

        hasher = hashlib.sha3_256()
        hasher.update(str.encode(p))
        phash = hasher.digest()

        hasher.update(str.encode(u))
        uhash = hasher.digest()

        found_user = users.query.filter_by(email=uhash).first()

        if found_user:
            if phash == found_user.pwh:
                session["username"] = found_user.email
                return(redirect(url_for("index")))
            else:
                return(redirect(url_for("sign_in")))
        else:
            return(redirect(url_for("sign_in")))

    return(redirect(url_for("sign_in")))

@app.route("/sign-in")
def sign_in():
    return render_template("sign_in.html")

@app.route("/add-user", methods=["GET", "POST"])
def add_user():

    if request.method == "POST":
        d = request.form.to_dict()

        u = d['uname']
        p = d['pwd']

        hasher = hashlib.sha3_256()
        hasher.update(str.encode(p))
        phash = hasher.digest()

        hasher.update(str.encode(u))
        uhash = hasher.digest()

        found_user = users.query.filter_by(email=uhash).first()

        if found_user:
            print("Invalid Email, already user with that email")
            return(redirect(url_for("sign_up")))
        else:

            usr = users(uhash, phash)
            db.session.add(usr)
            db.session.commit()
            return(redirect(url_for("sign_in")))

            
    else:
        return(redirect(url_for("sign_up")))

@app.route("/sign-up")
def sign_up():
    return render_template("sign_up.html")

@app.route("/logout", methods=["GET", "POST"])
def logout():
    if request.method == "POST":
        session['username'] = None
    return(redirect(url_for("index")))


@app.route("/set-session-model", methods=["GET", "POST"])
def set_session_model():
    if request.method == "POST":
        s = request.form.get("model-select")
        model = jmods.query.filter_by(created_by=session['username'], mname=s).first()

        if model:
            session['umodi'] = model._id
            return(redirect(url_for("model_inference")))
        return(redirect(url_for("mif")))
    return(redirect(url_for("mif")))


@app.route("/remove-model", methods=["GET", "POST"])
def remove_model():
    if request.method == 'POST':
        name = list(request.form.keys())[0]
        model = jmods.query.filter_by(created_by=session['username'], mname=name).first()
        db.session.delete(model)
        db.session.commit()
        return(redirect(url_for("model_dashboard")))
    return(redirect(url_for("model_dashboard")))

@app.route("/remove-user", methods=["GET", "POST"])
def remove_user():
    if request.method == "POST":
        username = users.query.filter_by(email=session['username']).first()
        db.session.delete(username)
        db.session.commit()
        return(redirect(url_for("sign_up")))
    return redirect(url_for("index"))

def gen_frames(mjson, mdt):
    camera = Camera(Annotator(mjson=mjson, mdt=mdt))
    while True:
        frame = camera.get_frame()

        print(type(frame))
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video-feed')
def video_feed():
    model = jmods.query.get(session['umodi']).first()
    return Response(gen_frames(model.model, model.mdt), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    db.create_all()
    app.run(port=80)