from gpiozero import DigitalInputDevice, DistanceSensor, Motor
from cv2 import waitKey

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


if __name__ == "__main__":
    allowed = {
        ord("w"): forward,
        ord("s"): stop,
        ord("l"): turnLeft,
        ord("r"): turnRight,
    }

    print("Allowed only: {}".format(allowed.keys()))

    while True:
        key = waitKey(0) & 0xFF

        print(key)

        if key in allowed.keys():
            allowed[key]()
        else:
            stop()
            exit(0)
