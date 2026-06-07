from flask import Flask, render_template, Response
from flask_socketio import SocketIO, send, emit
import cv2
import HandTrackingModule as htm
import math

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecret'

socketio = SocketIO(app, cors_allowed_origins="*")


def handTracking():  # generate frame by frame from camera
    wCam, hCam = 500, 300

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    detector = htm.handDetector(detectionCon=0.7)

    while True:
        success, img = cap.read()

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            length = math.hypot(x2 - x1, y2 - y1)
            # print(length)

            if 50.0 < length < 220.0:
                w = ((length - 50.0) / 170) * 20.0
            elif length >= 220.0:
                w = 20.0
            else:
                w = 0

            socketio.send(str(int(w)), broadcast=True)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        key = cv2.waitKey(1)
        if key == 27:
            break


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(handTracking(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    # app.run(debug=True)
    # host = your IP Address
    app.run(host='192.168.0.111', port=5000, debug=True)
    socketio.run(app)
