import cv2
import mediapipe as mp

capture = cv2.VideoCapture(0)
mpHandCapture = mp.solutions.hands
hands = mpHandCapture.Hands()

mpDraw = mp.solutions.drawing_utils

while True:
    success, img = capture.read()
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)

    h, w, c = img.shape

    if results.multi_hand_landmarks:
        for handLandMarks in results.multi_hand_landmarks:
 
            myHand = {}
            mylmList = []
            xList = []
            yList = []

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
            cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                            (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                            (255, 0, 255), 2)
    
    cv2.imshow("Sign Language Detector", img)
    cv2.waitKey(1)
