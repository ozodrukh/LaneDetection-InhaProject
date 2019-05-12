from gpiozero import DigitalInputDevice, DistanceSensor, Motor

import sys, cv2, time

rightIRSensor = DigitalInputDevice(12)
leftIRSensor = DigitalInputDevice(16)

# IR_tracer      8   L
#               7   R
rightIRTracerSensor = DigitalInputDevice(7)
leftIRTracerSensor = DigitalInputDevice(8)

# DisatanceSensor pin   21 T
#                      20 E
distanceSensor = DistanceSensor(26, 19)

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


if __name__ == "__main__":
    allowed = {
        ord("w"): forward,
        ord("s"): back,
        ord("a"): turnLeft,
        ord("d"): turnRight,
        ord("f"): stop
    }

    print("Allowed only: {}".format(allowed.keys()))

    camera = cv2.VideoCapture(0)

    while camera.isOpened():
        _, frame = camera.read()

        cv2.imshow("camera", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in allowed.keys():
            allowed[key]()
        elif key == ord("q"):
            stop()
            camera.release()
            cv2.destroyAllWindows()
            exit(0)

        time.sleep(0.01)
        stop()
