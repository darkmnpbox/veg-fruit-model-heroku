import os
from email.mime import image
from flask import Flask , render_template, redirect, request, flash, url_for
import numpy as np
from joblib import load
import cv2
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMAGE_SIZE = (224, 224)
IMAGE_LABELS = [
    'Apple','Bitter_Melon','Brinjal_Dotted','Chilli',
    'Fig','Green_Orange','Khira','Kiwi','Onion','Papper',
    'Pomogranate','Red_Cabbage','SMG','Sapodilla','Spoung_Gourd',
    'Straberry','Tomato_Green','Tomato_Red', 'Watermellon'
]
model = load('./veg.pki')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/result', methods=["post"])
def result():
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['image']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        resized_img = cv2.resize(img, IMAGE_SIZE)
        scaled_img = np.array(resized_img)/255.0
        result = np.argmax(model.predict(scaled_img[np.newaxis, ...]))
        obj = IMAGE_LABELS[result]
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template("result.html", obj=obj)
        # return redirect(url_for('download_file', name=filename))

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')