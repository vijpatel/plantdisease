#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pickle
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model

default_image_size = tuple((224, 224))
def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None

loaded_model = load_model('model.h5')

#label_encoder = open('D:\PlantVillage\Resnet50_model\label_transform.pkl', 'rb')
#label_binarizer = pickle.load(label_encoder)
prediction_classes = [['Pepper Bell','Bacterial Spot'],['Pepper Bell','Healthy'],
                      ['Potato','Early Blight'],['Potato','Late Blight'],
                      ['Potato','Healthy'],['Tomato','Bacterial Spot'],
                      ['Tomato','Early Blight'],['Tomato','Late Blight'],
                      ['Tomato','Leaf Mold'],['Tomato','Septoria Leaf Spot'],
                      ['Tomato','Spider Mites'],['Tomato','Target Spot'],
                      ['Tomato','Yellow Leaf Curl Virus'],['Tomato','Mosaic Virus'],
                      ['Tomato','Healthy']]
#labels = np.array(prediction_classes)[indices.astype(int)]

#print(labels)

def predict_disease(image_path):
    image_array = convert_image_to_array(image_path)
    np_image = np.array(image_array, dtype=np.float32)
    #print(np_image.dtype)
    np_image = np.expand_dims(np_image,0)
    #print(np_image.shape)
    #plt.imshow(plt.imread(image_path))
    #result = loaded_model.predict_classes(np_image)
    pred = loaded_model.predict(np_image)
    result = np.argmax(pred, axis=-1)
    percent = pred[0][result]    
    #pred = label_binarizer.classes_[result]
    #pred = labels[result]
    pred = np.array(prediction_classes)[result.astype(int)]
    return pred, percent
