import cv2 as cv
import mediapipe as mp

from keras.models import load_model
import numpy as np
import math

model = load_model('keras_model.h5', compile=False)
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

capture = cv.VideoCapture(0)
mpHandCapture = mp.solutions.hands
hands = mpHandCapture.Hands()

mpDraw = mp.solutions.drawing_utils

while True:
    success, img = capture.read()
    imageRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    h, w, c = img.shape

    if results.multi_hand_landmarks:
        for handLandMarks in results.multi_hand_landmarks:
            
            myHand = {}
            mylmList = []
            xList = []
            yList = []
            analysisframe = img

            for id, lm in enumerate(handLandMarks.landmark):
                px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                mylmList.append([px, py, pz])
                xList.append(px)
                yList.append(py)
            
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            boxW, boxH = xmax - xmin, ymax - ymin
            bbox = xmin, ymin, boxW, boxH
            cx, cy = bbox[0] + (bbox[2] // 2), \
                        bbox[1] + (bbox[3] // 2)

            myHand["lmList"] = mylmList
            myHand["bbox"] = bbox
            myHand["center"] = (cx, cy)

            
            mpDraw.draw_landmarks(img, handLandMarks, mpHandCapture.HAND_CONNECTIONS)
            
            imgWhite = np.ones((300 , 300, 3), np.uint8) * 255

            imgCrop = img[bbox[1] - 30:bbox[1] + bbox[3] + 30, bbox[0] - 30:bbox[0] + bbox[2] + 30]

            imgCropShape = img.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = 300 / h
                wCal = math.ceil(k*w)
                imgResize = cv.resize(imgCrop, (wCal, 300))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((300 - wCal)/2)
                imgWhite[:, wGap:wCal + wGap] = imgResize

                imgS = cv.resize(imgWhite, (224, 224))
                image_array = np.asarray(imgS)
                normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

                data[0] = normalized_image_array

                prediction = model.predict(data)
                index = np.argmax(prediction)
                class_name = letterpred[index]
                confidence_score = prediction[0][index]*100

            else:
                k = 300 / w
                hCal = math.ceil(k*h)
                imgResize = cv.resize(imgCrop, (300, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((300 - hCal)/2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

                imgS = cv.resize(imgWhite, (224, 224))
                image_array = np.asarray(imgS)
                normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

                data[0] = normalized_image_array

                prediction = model.predict(data)
                index = np.argmax(prediction)
                class_name = letterpred[index]
                confidence_score = prediction[0][index]*100

            output = str(class_name) + " "+ str(confidence_score)

            cv.putText(img, output, (bbox[0], bbox[1]-50), cv.FONT_ITALIC, 0.5, (51,255,51), 1)
            cv.rectangle(img, (bbox[0] - 30, bbox[1] - 30),
                        (bbox[0] + bbox[2] + 30, bbox[1] + bbox[3] + 30),
                        (255, 0, 255), 2)
            
            cv.imshow("Cropped Image", imgCrop)
            cv.imshow("White Image", imgWhite)


    cv.imshow("Sign Language Detector", img)

    cv.waitKey(1)

    if cv.getWindowProperty("Sign Language Detector",cv.WND_PROP_VISIBLE) < 1:        
        break      

cv.destroyAllWindows()