
# app.py
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
import numpy as np
from PIL import Image
import io
import cv2



app = Flask(__name__,template_folder='templates')
app.secret_key = 'password'

# Load the trained model
model = tf.keras.models.load_model('covid_prediction_model.h5')
print(model)

app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
def index():
    return render_template('predict.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    global glo_username
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

        filename=file.filename
            
        sample_image_path = 'static/uploads/'+filename


        img = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE)
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        clarity_value=laplacian_var
        threshold=300

        if clarity_value < threshold:
            sample_image = tf.keras.preprocessing.image.load_img(sample_image_path, target_size=(224, 224))
            sample_image = tf.keras.preprocessing.image.img_to_array(sample_image)
            sample_image = sample_image / 255.0
            sample_image = tf.expand_dims(sample_image, 0)
            prediction = model.predict(sample_image)
            print('Prediction:', prediction)

            result = "COVID-19 Positive" if prediction[0][0] > 0.5 else "COVID-19 Negative"
            print(result)
        
            return render_template('predict.html', prediction=result, image_file=filename)
        return render_template('predict.html',prediction="Invalid Image  Please Upload X-Ray Image Only..!")
    
    return render_template('predict.html',predict="Invalid Image")


if __name__ == "__main__":
    app.run(debug=True)