import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

photos_path = 'photos/'
source_photos_path = photos_path + 'source/'
output_photos_path = photos_path + 'output/'

source_photos = os.listdir(source_photos_path)
output_photos = os.listdir(output_photos_path)

photos_to_pass = [i for i in source_photos if i not in output_photos]


def main(photos_to_pass):
    for i in photos_to_pass:
        print(source_photos_path+i)
        img = cv2.imread(source_photos_path+i)
        output = detect_faces(img)
        try:
            cv2.imwrite(output_photos_path+i, output, [cv2.COLOR_BGR2RGB])
        except Exception:
            print('No faces detected')


def detect_faces(img):
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_detector.detectMultiScale(grey_img, scaleFactor=1.1, minNeighbors=10, minSize=(40,40))
    output = apply_to_region(img, faces)
    return output

def apply_to_region(img, faces):
    for (x, y, w, h) in faces:
        roi = img[y:y+h, x:x+w, :]
        blur_roit = apply_blur(roi)
        img[y:y+h, x:x+w, :] = blur_roit
        return img
        
def apply_blur(roi):
    kernel_size = 70
    kernel_h = create_kernel(kernel_size)
    roi = cv2.filter2D(roi, -1, kernel_h)
    return roi

def create_kernel(kernel_size):
    kernel_h = np.zeros((kernel_size, kernel_size))
    kernel_h[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel_h /= kernel_size
    return kernel_h

main(photos_to_pass)