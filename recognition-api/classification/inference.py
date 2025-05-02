import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Set this to the number of cores you want to use

import cv2

from classification.feature_extraction import extract_features

class_license_plate = {0: 'private', 1: 'light_private', 2: 'commerce', 3: 'light_commerce', 4: 'designed'}

def inference_class(img, model, scaler):

    # Read and preprocess the image
    img = cv2.resize(img, (100, 50))
    features = extract_features(img).reshape(1, -1)

    features = scaler.transform(features)

    # Predict the class
    prediction = model.predict(features)
    predicted_label = class_license_plate[prediction[0]]

    return predicted_label
