import cv2
import mediapipe as mp
import time
import keyboard
import webbrowser


class handDetector():
    # def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
    def __init__(self, mode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,self.detectionCon,self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            thumb_tip, index_tip = None, None

            for id, lm in enumerate(myHand.landmark):
                # Detect thumb tip and index finger tip landmarks
                if id == 4:  # Thumb tip
                    thumb_tip = lm
                elif id == 8:  # Index finger tip
                    index_tip = lm

                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)

            # Check if thumb and index finger tips are close enough to trigger button press
            if thumb_tip and index_tip and (abs(thumb_tip.x - index_tip.x) < 0.03) and (
                    abs(thumb_tip.y - index_tip.y) < 0.03):
                keyboard.press('space')
                keyboard.release('space')

        return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if len(lmList) !=0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
url = "https://projects.gauracs.com/dinogame/"
webbrowser.open(url)

if __name__ == "__main__":
    main()