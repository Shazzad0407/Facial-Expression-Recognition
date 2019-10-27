# import the necessary packages
import keras
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
import base64
from flask import request, render_template, Flask
import cv2
from keras.models import model_from_json
import re,os
from werkzeug.utils import secure_filename
import uuid

# initialize our Flask application and the Keras model
model = None
full_path=None

def create_app():
    app = Flask(__name__)

    def load_model():
        print("hello")
        global model
        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("model.h5")
        print("Loaded model from disk")
        model.compile(loss='categorical_crossentropy',
                      optimizer="rmsprop",
                      metrics=['accuracy'])
    load_model()
    return app


app = create_app()

#
# def prepare_image(image, target):
#     # if the image mode is not RGB, convert it
#     if image.mode == "RGB":
#         image = image.convert("L")
#
#     # resize the input image and preprocess it
#     image = image.resize(target)
#     image = img_to_array(image)
#     image = image/255.0
#     image = image.reshape(1, target[0], target[1], 1)
#     print(image.shape)
#     # return the processed image
#     return image
def do_prediction(image):
    global full_path
    label_map = ["Angry", "Fear", "Happy",
                 "Sad", "Surprise", "Neutral"]
    im = cv2.imread(image)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("cascade_frontface.xml")
    faces = face_cascade.detectMultiScale(gray,1.2, 3, minSize=(80, 80))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2, 5)
        face_crop = im[y:y + h, x:x + w]
        face_crop = cv2.resize(face_crop, (48, 48))
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face_crop = face_crop.astype('float32') / 255.0
        face_crop = np.asarray(face_crop)
        face_crop = face_crop.reshape(
            1, 1, face_crop.shape[0], face_crop.shape[1])
        result = label_map[np.argmax(model.predict(face_crop))]
        cv2.putText(im, result, (x, y), font, 1, (200, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('result', im)
    random_value=str(uuid.uuid4())
    new_image=random_value+".jpg"

    if not os.path.exists("/predicted_images/directory"):
        os.makedirs("/predicted_images")
    app.config["UPLOAD_FOLDER"] = "/predicted_images"
    full_path=os.path.join(app.config['UPLOAD_FOLDER'], new_image)
    print(full_path)
    cv2.imwrite(full_path, im)

@app.route("/", methods=["GET"])
def index():
    return render_template("predict.html")


@app.route("/predict", methods=["POST"])
def predict():
        # initialize the data dictionary that will be returned from the
        # view
    data = {"success": False}
    message = request.get_json(force=True)
    encoded = message["image"]
    encoded = re.sub('^data:image/.+;base64,', '', encoded)
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # classify the input image and then initialize the list
    # of predictions to return to the client

    do_prediction(opencvImage)
    # indicate that the request was a success
    data["success"] = True
    data["image"]=full_path

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == "__main__":
    app.run()
