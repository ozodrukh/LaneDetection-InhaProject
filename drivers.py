from gpiozero import DigitalInputDevice, DistanceSensor, Motor
from functools import partial as bind
import datetime

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


def forward(left=motor_configs["forward.x"],
            right=motor_configs["forward.y"]):
    sensors["motor"]["left"].forward(left)
    sensors["motor"]["right"].forward(right)


def back(left=motor_configs["back.x"],
         right=motor_configs["back.y"]):
    sensors["motor"]["left"].backward(left)
    sensors["motor"]["right"].backward(right)


def turn_left(x=motor_configs["left.x"],
              y=motor_configs["left.y"]):
    sensors["motor"]["left"].forward(x)
    sensors["motor"]["right"].backward(y)


def turn_right(back=motor_configs["right.x"],
               forward=motor_configs["right.y"]):
    sensors["motor"]["left"].backward(back)
    sensors["motor"]["right"].forward(forward)


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


if __name__ == "__main__":
    import cv2
    import os, time

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

    action_time = 0
    action_timeout = 220

    while camera.isOpened():
        _, frame = camera.read()

        video_output.write(frame)
        cv2.imshow("camera", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in allowed.keys():
            action_time = datetime.datetime.now()
            allowed[key]()

        elif key == ord("q"):
            stop()
            camera.release()
            video_output.release()
            cv2.destroyAllWindows()
            exit(0)

        if action_time > 0 and (datetime.datetime.now() - action_time).microseconds > action_timeout:
            stop()
