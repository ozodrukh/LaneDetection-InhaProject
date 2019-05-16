import datetime
from functools import partial as bind

from gpiozero import DigitalInputDevice, DistanceSensor, Motor

sensors = {
    "ir": {
        "left": DigitalInputDevice(12),
        "right": DigitalInputDevice(16)
    },
    "trace": {
        "left": DigitalInputDevice(7),
        "right": DigitalInputDevice(8)
    },
    "distance": DistanceSensor(26, 19),
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

    # print("left:{}, right={}".format(left, right))

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

    # if abs(angel) < 65:
    #     if angel > 0:
    #         if angel < 45:
    #             x = angel / 140
    #         else:
    #             x = angel / 70
    #         motor_configs["forward.y"] = 0.85
    #         motor_configs["forward.x"] = x
    #         forward()
    #         return
    #     else:
    #         if angel > -45:
    #             y = abs(angel / 140)
    #         else:
    #             y = abs(angel / 70)
    #
    #         motor_configs["forward.y"] = y
    #         motor_configs["forward.x"] = 0.85
    #         forward()
    #         return
    # if abs(angel) < 55:
    if angel > 0:
        if angel > 15 and angel < 20:
            motor_configs["forward.y"] = 1
            motor_configs["forward.x"] = 0.10
        elif angel > 20 and angel < 25:
            motor_configs["forward.y"] = 1
            motor_configs["forward.x"] = 0.13
        elif angel > 25 and angel < 30:
            motor_configs["forward.y"] = 1
            motor_configs["forward.x"] = 0.16
        elif angel < 30 and angel < 35:
            motor_configs["forward.y"] = 1
            motor_configs["forward.x"] = 0.20
        elif angel < 35 and angel < 40:
            motor_configs["forward.y"] = 1
            motor_configs["forward.x"] = 0.25
        elif angel < 40 and angel < 45:
            motor_configs["forward.y"] = 1
            motor_configs["forward.x"] = 0.28
        elif angel < 45 and angel < 50:
            motor_configs["forward.y"] = 1
            motor_configs["forward.x"] = 0.3
        elif angel < 50 and angel < 55:
            motor_configs["forward.y"] = 1
            motor_configs["forward.x"] = 0.35
        else:
            motor_configs["forward.y"] = 0.55
            motor_configs["forward.x"] = 0.5

    else:
        if angel < -15 and angel < -20:
            motor_configs["forward.y"] = 0.10
            motor_configs["forward.x"] = 1
        elif angel < -20 and angel < -25:
            motor_configs["forward.y"] = 0.13
            motor_configs["forward.x"] = 1
        elif angel < -25 and angel < -30:
            motor_configs["forward.y"] = 0.16
            motor_configs["forward.x"] = 1
        elif angel < -30 and angel < -35:
            motor_configs["forward.y"] = 0.22
            motor_configs["forward.x"] = 1

        elif angel < -35 and angel < -40:
            motor_configs["forward.y"] = 0.25
            motor_configs["forward.x"] = 1

        elif angel < -40 and angel < -45:
            motor_configs["forward.y"] = 0.28
            motor_configs["forward.x"] = 1
        elif angel < -45 and angel < -50:
            motor_configs["forward.y"] = 0.3
            motor_configs["forward.x"] = 1
        elif angel < -50 and angel < -55:
            motor_configs["forward.y"] = 0.35
            motor_configs["forward.x"] = 1

        else:
            motor_configs["forward.y"] = 0.5
            motor_configs["forward.x"] = 0.55
    forward()
    # return
    # motor_configs["forward.x"] = 0.5
    # motor_configs["forward.y"] = 0.5
    # forward()


def main():
    import cv2
    import os

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
