import datetime
import os
from functools import partial as bind

import cv2
from gpiozero import DigitalInputDevice, Motor

sensors = {
    "ir": {
        "left": DigitalInputDevice(12),
        "right": DigitalInputDevice(16)
    },
    "trace": {
        "left": DigitalInputDevice(7),
        "right": DigitalInputDevice(8)
    },
    # "distance": DistanceSensor(26, 19),
    "motor": {
        'left': Motor(18, 23),
        'right': Motor(24, 25)
    }
}
motor_configs = {}

for side in ["left", "right", "forward", "back"]:
    motor_configs[side + ".x"] = 1
    motor_configs[side + ".y"] = 1

forwardPin = False


def get_distance():
    return sensors["distance"].distance


def forward():
    left = motor_configs["forward.y"]
    right = motor_configs["forward.x"]

    print("left:{}, right={}".format(left, right))

    sensors["motor"]["left"].forward(left)
    sensors["motor"]["right"].forward(right)


def back():
    left = motor_configs["back.x"]
    right = motor_configs["back.y"]

    sensors["motor"]["left"].backward(left)
    sensors["motor"]["right"].backward(right)


def turn_left():
    x = motor_configs["left.x"]
    y = motor_configs["left.y"]

    sensors["motor"]["left"].forward(x)
    sensors["motor"]["right"].backward(y)


def turn_right():
    x = motor_configs["right.x"]
    y = motor_configs["right.y"]

    sensors["motor"]["left"].backward(x)
    sensors["motor"]["right"].forward(y)


def stop():
    sensors["motor"]["left"].stop()
    sensors["motor"]["right"].stop()


def lIRsensor():
    return sensors["ir"]["left"].value


def rIRsensor():
    return sensors["ir"]["right"].value


def rIRTracerSensor():
    return sensors["trace"]["left"].value


def lIRTracerSensor():
    return sensors["trace"]["left"].value


def on_road_detected(direction, angel):
    print(direction, angel)

    if abs(angel) < 90:
        if angel > 0:
            if 15 < angel < 20:
                motor_configs["forward.y"] = 1
                motor_configs["forward.x"] = 0.10
            elif 20 < angel < 25:
                motor_configs["forward.y"] = 1
                motor_configs["forward.x"] = 0.13
            elif 25 < angel < 30:
                motor_configs["forward.y"] = 1
                motor_configs["forward.x"] = 0.16
            elif 30 > angel > 35:
                motor_configs["forward.y"] = 1
                motor_configs["forward.x"] = 0.20
            elif 35 > angel > 40:
                motor_configs["forward.y"] = 1
                motor_configs["forward.x"] = 0.22
            elif 40 > angel > 45:
                motor_configs["forward.y"] = 1
                motor_configs["forward.x"] = 0.25
            elif 45 > angel > 50:
                motor_configs["forward.y"] = 1
                motor_configs["forward.x"] = 0.28
            elif 50 > angel > 55:
                motor_configs["forward.y"] = 1
                motor_configs["forward.x"] = 0.30
            elif 55 > angel > 60:
                motor_configs["forward.y"] = 0.6
                motor_configs["forward.x"] = 0.4
            elif 60 > angel > 65:
                motor_configs["forward.y"] = 0.6
                motor_configs["forward.x"] = 0.5
            elif 65 > angel > 70:
                motor_configs["forward.y"] = 0.6
                motor_configs["forward.x"] = 0.57
            elif 70 > angel > 75:
                motor_configs["forward.y"] = 0.57
                motor_configs["forward.x"] = 0.6
            elif 75 > angel > 80:
                motor_configs["forward.y"] = 0.55
                motor_configs["forward.x"] = 0.6
            elif 80 > angel > 90:
                motor_configs["forward.y"] = 0.5
                motor_configs["forward.x"] = 0.6
            else:
                motor_configs["forward.y"] = 0.6
                motor_configs["forward.x"] = 0.6
        else:
            if -15 > angel > -20:
                motor_configs["forward.y"] = 0.10
                motor_configs["forward.x"] = 1
            elif -20 > angel > -25:
                motor_configs["forward.y"] = 0.13
                motor_configs["forward.x"] = 1
            elif -25 > angel > -30:
                motor_configs["forward.y"] = 0.16
                motor_configs["forward.x"] = 1
            elif -30 > angel > -35:
                motor_configs["forward.y"] = 0.20
                motor_configs["forward.x"] = 1
            elif -35 > angel > -40:
                motor_configs["forward.y"] = 0.22
                motor_configs["forward.x"] = 1
            elif -40 > angel > -45:
                motor_configs["forward.y"] = 0.25
                motor_configs["forward.x"] = 1
            elif -45 > angel > -50:
                motor_configs["forward.y"] = 0.30
                motor_configs["forward.x"] = 1
            elif -50 > angel > -55:
                motor_configs["forward.y"] = 0.35
                motor_configs["forward.x"] = 1
            elif -55 > angel > -60:
                motor_configs["forward.y"] = 0.35
                motor_configs["forward.x"] = 0.6
            elif -60 > angel > -65:
                motor_configs["forward.y"] = 0.4
                motor_configs["forward.x"] = 0.6
            elif -65 > angel > -70:
                motor_configs["forward.y"] = 0.57
                motor_configs["forward.x"] = 0.6
            elif -70 > angel > -75:
                motor_configs["forward.y"] = 0.6
                motor_configs["forward.x"] = 0.57
            elif -75 > angel > -80:
                motor_configs["forward.y"] = 0.6
                motor_configs["forward.x"] = 0.5
            elif -80 > angel > -85:
                motor_configs["forward.y"] = 0.6
                motor_configs["forward.x"] = 0.47
            elif -85 > angel > -90:
                motor_configs["forward.y"] = 0.6
                motor_configs["forward.x"] = 0.4
            else:
                motor_configs["forward.y"] = 0.5
                motor_configs["forward.x"] = 0.5
    if sensors["tracer"]["left"].value == 1 and motor_configs["forward.x"] < 1:
        motor_configs["forward.x"] += 1
    if sensors["tracer"]["right"].value == 1 and motor_configs["forward.y"] < 1:
        motor_configs["forward.y"] += 1
    forward()


def main():
    width = 640
    height = 480

    if not os.path.exists('/dev/video0'):
        path = 'sudo modprobe bcm2835-v4l2 max_video_width=640 max_video_height=480'
        os.system(path)

    allowed = {
        ord("w"): forward,
        ord("s"): back,
        ord("a"): turn_left,
        ord("d"): turn_right,
        ord("f"): stop
    }

    print("Allowed only: {}".format(allowed.keys()))

    cv2.namedWindow("camera")

    camera = cv2.VideoCapture(0)
    camera.set(3, width)
    camera.set(4, height)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_output = cv2.VideoWriter('./session.avi', fourcc, 30.0, (640, 480))

    def set_motor_config(config_name, value):
        motor_configs[config_name] = value / 100

    for config in motor_configs:
        update_config = bind(set_motor_config, config)

        cur_value = motor_configs[config] * 100
        cv2.createTrackbar(config, "camera", cur_value, 100, update_config)
        cv2.setTrackbarMin(config, "camera", 0)
        cv2.setTrackbarMax(config, "camera", 100)

    action_time = None
    action_timeout = 250

    direction = []

    while camera.isOpened():
        _, frame = camera.read()

        video_output.write(frame)
        cv2.imshow("camera", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in allowed.keys():
            action_time = datetime.datetime.now()

            if key not in direction:
                direction.append(key)
                allowed[key]()

            print(allowed[key])

        elif key == ord("q"):
            stop()
            camera.release()
            video_output.release()
            cv2.destroyAllWindows()
            exit(0)

        if action_time is not None and (datetime.datetime.now() - action_time).microseconds / 1000 > action_timeout:
            stop()
            print("stop")
            direction.clear()
            action_time = None


if __name__ == "__main__":
    # max()
    pass
