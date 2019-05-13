from gpiozero import DigitalInputDevice, DistanceSensor, Motor
from functools import partial as bind

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

forwardPin = False


def get_distance():
    return sensors["distance"].distance


def forward(left=1.0, right=1.0):
    sensors["motor"]["left"].forward(left)
    sensors["motor"]["right"].forward(right)


def back(left=1.0, right=1.0):
    sensors["motor"]["left"].backward(left)
    sensors["motor"]["right"].backward(right)


def turn_left(x=1.0, y=1.0):
    sensors["motor"]["left"].forward(x)
    sensors["motor"]["right"].backward(y)


def turn_right(back=1.0, forward=1.0):
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


motor_configs = {}

for side in ["left", "right", "forward", "back"]:
    motor_configs[side + ".x"] = 1
    motor_configs[side + ".y"] = 1

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

    def set_motor_config(config_name, value):
        motor_configs[config_name] = value / 100

    for config in motor_configs:
        update_config = bind(set_motor_config, config)

        cur_value = motor_configs[config] * 100
        cv2.createTrackbar(config, "camera", cur_value, 100, update_config)
        cv2.setTrackbarMin(config, "camera", 0)
        cv2.setTrackbarMax(config, "camera", 100)

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

        time.sleep(0.1)
        stop()
