import cv2
import os
import numpy as np
capture = cv2.VideoCapture(0)

photos_path = 'photos/'
source_photos_path = photos_path + 'source/'
output_photos_path = photos_path + 'output/'

source_photos = os.listdir(source_photos_path)
output_photos = os.listdir(output_photos_path)

photos_to_pass = [i for i in source_photos]
face_detector = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')


def main():
    img = capture.read()[1]

    w, h = img.shape[:2]
    output = detect_faces(img, w, h)
    cv2.imshow('frame', output)


def detect_faces(img, w, h):
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (400, 400)))
    face_detector.setInput(blob)
    detections = face_detector.forward()
    output = apply_to_region(img, detections, w, h)
    return output


def apply_to_region(img, detections, w, h):
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.99:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            if any(box < 0):
                break
            x, y, w, h = box.astype('int')
            roi = img[y:h, x:x+w, :]
            if not roi.tolist():
                break
            blur_roit = apply_blur(roi)
            img[y:h, x:x+w, :] = blur_roit
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


while True:
    main()
    if cv2.waitKey(1) == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
