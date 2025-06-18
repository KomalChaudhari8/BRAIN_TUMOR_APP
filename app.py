from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import backend as K
import numpy as np
import os

app = Flask(__name__)

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed

model = load_model(
    'model/mobilenetv2_brain_tumor_classifier.keras',
    custom_objects={'focal_loss_fixed': focal_loss(gamma=2., alpha=0.25)}
)

classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if file:

        for f in os.listdir('static'):
            if f.endswith(('.png', '.jpg', '.jpeg')):
                os.remove(os.path.join('static', f))

        img_path = os.path.join('static', file.filename)
        file.save(img_path)

        img = load_img(img_path, target_size=(224, 224), color_mode='grayscale')
        img_arr = img_to_array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        prediction = model.predict(img_arr)
        predicted_class = classes[np.argmax(prediction)]

        return render_template('result.html', prediction=predicted_class, image_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
