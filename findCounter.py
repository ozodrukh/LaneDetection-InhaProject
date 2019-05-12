import io
import sys
import time

import cv2
import numpy as np
from gpiozero import DigitalInputDevice, DistanceSensor, Motor


def main():
    pi = np.math.pi
    i = 0
    rightIRSensor = DigitalInputDevice(12)
    leftIRSensor = DigitalInputDevice(16)

    # IR_tracer      8   L
    #               7   R
    rightIRTracerSensor = DigitalInputDevice(7)
    leftIRTracerSensor = DigitalInputDevice(8)

    # DisatanceSensor pin   21 T
    #                      20 E
    distanceSensor = DistanceSensor(21, 20)

    # Motors
    motor_left = Motor(18, 23)
    motor_right = Motor(24, 25)

    forwardpin = False

    def getDistance():
        return distanceSensor.distance

    def forward(left=1.0, right=1.0):
        motor_left.forward(left)
        motor_right.forward(right)

    def back(left=1.0, right=1.0):
        motor_left.backward(left)
        motor_right.backward(right)

    def turnRight(x=1.0, y=1.0):
        motor_left.forward(x)
        motor_right.backward(y)

    def turnLeft(back=1.0, forward=1.0):
        motor_left.backward(back)
        motor_right.forward(forward)

    def stop():
        motor_left.stop()
        motor_right.stop()

    def lIRsensor():
        return rightIRSensor.value

    def rIRsensor():
        return leftIRSensor.value

    def rIRTracerSensor():
        return rightIRTracerSensor.value

    def lIRTracerSensor():
        return leftIRTracerSensor.value

    def saveframe(name, frame):
        cv2.imwrite(str(name) + '/png', frame)

    def halfRightTurn():
        turnRight(1, 0.5)
        time.sleep(0.6)
        stop()

    def hougeline(frame, i=0):
        kernel = np.ones((3, 3), np.uint8)
        localfram = frame
        # frame = cv2.dilate(frame, kernel,iterations=3)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # blur = cv.GaussianBlur(frame, (5, 5), 0)
        blur = cv2.medianBlur(frame, 5)
        canny = cv2.Canny(blur, 150, 150, apertureSize=3)
        cv2.imshow('canny', canny)
        lines = cv2.HoughLines(canny, 1, pi / 180, 50)  # srn=80, stn=200)

        if lines is not None:
            print("Len of lines:", len(lines))

        if lines is not None:
            # sumtheta = 0
            sumx1 = 0
            sumx2 = 0
            sumy1 = 0
            sumy2 = 0
            counter = len(lines)
            for line in lines:
                # print(line)

                rho, theta = line[0]
                # print(theta/pi*180)
                if line is not None:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    scalar = 1000
                    x1 = int(x0 + scalar * (-b))
                    y1 = int(y0 + scalar * (a))

                    x2 = int(x0 - scalar * (-b))
                    y2 = int(y0 - scalar * (a))
                    if abs(y2 - y1) < scalar / 10:
                        counter -= 1
                        continue
                    if abs(x2 - x1) < scalar / 20:
                        counter -= 1
                        continue
                    sumy1 += y1
                    sumx1 += x1
                    sumx2 += x2
                    sumy2 += y2
                    cv2.line(localfram, (x1, y1), (x2, y2), (0, 0, 255), 2, 4)

            try:
                dely = (sumy2 - sumy1) / counter
                delx = (sumx2 - sumx1) / counter

                cv2.line(localfram, (int(sumx1 / counter), int(sumy1 / counter)),
                         (int(sumx2 / counter), int(sumy2 / counter)), (0, 255, 0), 2, 4)
                print('dely=', dely)
                print('delx=', delx)
                font = cv2.FONT_HERSHEY_PLAIN
                cv2.putText(localfram, str(dely), (0, 50), font, 4, (255, 0, 0), 4)
                cv2.putText(localfram, str(delx), (0, 100), font, 4, (255, 255, 0), 4)
                angel = (np.math.asin(dely / delx) / (pi / 2) * 180)
                cv2.putText(localfram, str(angel), (0, 150), font, 4, (0, 255, 255), 4)

                i += 1

                """WRITE TO FILE"""
                # cv2.imwrite((str(i) + '.png'), localfram)
                print(str(i) + '.png')
            except OSError as err:
                print("OS error: {0}".format(err))
            except ValueError:
                print("Could not convert data to an integer.")
            except:
                print("Unexpected error:", sys.exc_info()[0])
            # time.sleep(0.1)

        cv2.imshow('output', localfram)
        return localfram, i

    def hougelinep(frame, i=0):
        kernel = np.ones((3, 3), np.uint8)
        localfram = frame
        # frame = cv2.dilate(frame, kernel,iterations=3)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # blur = cv.GaussianBlur(frame, (5, 5), 0)
        canny = cv2.Canny(frame, 200, 150, apertureSize=3)
        cv2.imshow('canny', canny)
        lines = cv2.HoughLinesP(canny, 1, pi / 180, 10, minLineLength=5, maxLineGap=5)  # srn=80, stn=200)

        if lines is not None:
            print("Len of lines:", len(lines))

        if lines is not None:
            sumx1 = 0
            sumx2 = 0
            sumy1 = 0
            sumy2 = 0
            counter = len(lines)
            for line in lines:
                if line is not None:
                    x1, y1, x2, y2 = line[0]
                    sumy1 += y1
                    sumx1 += x1
                    sumx2 += x2
                    sumy2 += y2
                    cv2.line(localfram, (x1, y1), (x2, y2), (0, 0, 255), 2, 4)
            try:
                cv2.line(localfram, (int(sumx1 / counter), int(sumy1 / counter)),
                         (int(sumx2 / counter), int(sumy2 / counter)), (0, 255, 0), 2, 4)
                i += 1
            except OSError as err:
                print("OS error: {0}".format(err))
            except ValueError:
                print("Could not convert data to an integer.")
            except:
                print("Unexpected error:", sys.exc_info()[0])

        # print('delx=', delx)

        # print(sumtheta / counter)
        # time.sleep(0.5)
        # pprint.pprint(lines)
        # time.sleep(1)
        time.sleep(0.1)
        cv2.imshow('output', localfram)
        return localfram, i

    # The video feed is read in as a VideoCapture object
    # cap = cv2.VideoCapture("outcpp.avi")
    red_val = 0
    green_val = 65

    stream = io.BytesIO()
    my_file = open('my_image.jpg', 'wb')

    cap = cv2.VideoCapture(0)
    with picamera.PiCamera() as camera:
        camera.resolution = (1920, 1080)
        camera.framerate = 32
        camera.start_preview()
        time.sleep(2)
        # camera.capture(my_file)
        camera.capture(stream, format='jpeg')
        # define range of RED color in HSV
    halfRightTurn()

    while (True):
        # Capture frame-by-frame
        data = np.fromstring(stream.getvalue(), dtype=np.uint8)
        ret, frame = cap.read(data)

        try:
            frame2 = frame
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            width, height = frame.shape[:2]
            frame = frame[int(width / 2):width, 0:height]
        except:
            continue

        hougelinefram, i = hougeline(frame, i)
        cv2.imshow('original', frame2)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.waitKey(0)
        if cv2.waitKey(10) & 0xFF == ord('w'):
            print('w')
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    cap.release()


# cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    #