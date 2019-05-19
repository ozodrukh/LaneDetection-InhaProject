from configs import LOCAL_MODE, GUI_MODE, params

import configs
import cv2
import numpy as np
import functools, route

from datetime import datetime

if configs.PRODUCTION_MODE or LOCAL_MODE == False:
    import drivers

# sudo modprobe bcm2835-v4l2 # to enable camera on pi

size = configs.resize

LINE_COLOR_RANGE_LOWER = np.array([16, int(0.1 * 255), int(0.1 * 255)])
LINE_COLOR_RANGE_HIGHER = np.array([35, int(1 * 255), int(0.6 * 255)])

LINE_COLOR_RGB_RANGE_LOWER = np.array([0, 24, 28])
LINE_COLOR_RGB_RANGE_HIGHER = np.array([80, 169, 220])

direction = {
    "turn": None,
    "angel": None
}


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def render(frame):
    original = frame
    target_frame = cv2.medianBlur(original, 5)

    structure = cv2.getStructuringElement(cv2.MORPH_RECT, (params['kernel_width'], params['kernel_height']))

    if params['morph'] == "erode":
        morphed = cv2.erode(target_frame, structure)
    else:
        morphed = cv2.morphologyEx(target_frame, cv2.MORPH_GRADIENT, structure)

    original = morphed

    # morphed = cv2.cvtColor(morphed, cv2.COLOR_BGR2HSV)

    morphed = cv2.inRange(
        morphed,
        LINE_COLOR_RGB_RANGE_LOWER,
        LINE_COLOR_RGB_RANGE_HIGHER
    )

    # find_lines_using_Canny_and_HoughLines(original, morphed)
    find_lines_using_contours(original, morphed)

    if GUI_MODE:
        cv2.imshow("frame", original)
    # cv2.imshow("morphed", morphed)
    return original


def find_lines_using_contours(frame, morphed):
    if not configs.PRODUCTION_MODE:
        contours, _ = cv2.findContours(morphed, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_TREE)
    else:
        _, contours, _ = cv2.findContours(morphed, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_TREE)

    if contours is None:
        print("no contours")

        if GUI_MODE:
            cv2.putText(frame,
                        "No Contours",
                        (int(size[0] / 2), 0),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 255, 255))
        return

    if GUI_MODE:
        max_length = 0

        for i in range(len(contours)):
            length = cv2.arcLength(contours[i], False)
            if length > max_length:
                max_length = length

        for i in range(len(contours)):
            length = cv2.arcLength(contours[i], False)
            if length < 150:
                continue

            cv2.drawContours(frame, contours, i, (20, int(255 * length / max_length), int(255 * length / max_length)))

    t, a = route.find_lines_on_contours(frame, contours)

    direction["turn"] = t
    direction["angel"] = a


def find_lines_using_Canny_and_HoughLines(frame, morphed):
    morphed = cv2.Canny(morphed, 100, 200)

    lines = cv2.HoughLinesP(
        image=morphed,
        rho=params['hough.rho'],
        theta=params['hough.theta'],
        threshold=params['hough.threshold'],
        minLineLength=params['hough.min_distance'],
        maxLineGap=params['hough.max_line_gap']
    )

    route.find_lines_on_hough_lines(frame, lines)


def morph_type_changed(morph_type):
    if morph_type == 0:
        params['morph'] = "erode"
    else:
        params['morph'] = "dilate"

    if params['current_frame'] is not None:
        render(params['current_frame'])


def erode_changed(v):
    params['kernel_width'] = v

    if params['current_frame'] is not None:
        render(params['current_frame'])


def normalize_frame_for_lane_detection(frame):
    if frame.shape[0] != 300 and frame.shape[1] != 300:
        frame = cv2.resize(frame, size, cv2.INTER_NEAREST)

    bounds = [
        int(3 / 5 * size[1]), 0,
        size[0], size[1]
    ]

    frame = frame[bounds[0]:bounds[2], bounds[1]:bounds[3]]

    params['kernel_max_width'] = size[0]
    params['kernel_width'] = int(params['kernel_max_width'] / 40)

    params['current_frame'] = frame
    return frame


def main():
    camera = cv2.VideoCapture(configs.camera_target)

    if not camera.isOpened():
        print("Video not opened")
        return

    if GUI_MODE:
        cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)
        # cv2.namedWindow("morphed", cv2.WINDOW_AUTOSIZE)

        cv2.moveWindow("frame", 0, 0)
        # cv2.moveWindow("morphed", 400, 0)

        # attach_options_bar()
        # create_hough_line_editor()

    last_start = datetime.now()

    fps = 0
    current_fps = 0

    video_output = None

    if configs.CAPTURE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        video_output = cv2.VideoWriter('./session.avi', fourcc, 30.0, (640, 480))

    while camera.isOpened():
        _, frame = camera.read()

        if frame.size == 0:
            break

        if configs.CAPTURE_VIDEO and video_output is not None:
            video_output.write(frame)

        # frame = rotateImage(frame, -90)
        #

        # cv2.imshow("window", frame)
        frame = normalize_frame_for_lane_detection(frame)

        frame = render(frame)

        send_motor_signal()

        if (datetime.now() - last_start).seconds > 1:
            last_start = datetime.now()
            current_fps = fps
            fps = 0
        elif params["fps_counter"] and GUI_MODE:
            fps = fps + 1

            cv2.putText(frame, "fps: {}".format(current_fps),
                        (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

            cv2.imshow("frame", frame)

        try:
            key = cv2.waitKey(1)

            if key & 0xFF == ord('q'):
                break

            if key == 32:
                params["paused"] = True
                send_motor_signal()

                if cv2.waitKey(0) == 32:
                    params["paused"] = False
                    continue
        except KeyboardInterrupt:
            pass

    if video_output is not None:
        video_output.release()

    camera.release()


def send_motor_signal():
    if not configs.PRODUCTION_MODE:
        return

    if params["paused"]:
        drivers.stop()
        return

    drivers.on_road_detected(direction["turn"], direction["angel"])


def attach_options_bar():
    cv2.createTrackbar("morph_type", "frame", 0 if params['morph'] == 'erode' else 1, 1, morph_type_changed)
    cv2.createTrackbar("kernel_width", "frame", params['kernel_width'], size[0], erode_changed)

    cv2.setTrackbarMin("kernel_width", "frame", 1)


def create_hough_line_editor():
    def update(key, value):
        if key == "hough.theta":
            params[key] = np.pi / max(value, 1)
        else:
            params[key] = value

        render(params['current_frame'])

    for key in params.keys():
        if key.startswith("hough."):
            if key == "hough.theta":
                max_value = 360
                value = 180
            else:
                max_value = 200
                value = params[key]

            changed = functools.partial(update, key)
            cv2.createTrackbar(key, "frame", value, max_value, changed)
    pass


def create_rgb_bar():
    def trigger_changes(upper, i, x):
        target = LINE_COLOR_RGB_RANGE_HIGHER if upper else LINE_COLOR_RGB_RANGE_LOWER

        for index in range(3):
            if index == i:
                target[index] = x
            else:
                target[index] = target[index]

        render(params['current_frame'])

    def createRGBTrackbar(upper: bool):
        for i, key in enumerate(['b', 'g', 'r']):
            min = (LINE_COLOR_RGB_RANGE_HIGHER if upper else LINE_COLOR_RGB_RANGE_LOWER)[i]
            max = 255

            id = "upper" if upper else "lower"
            change = functools.partial(trigger_changes, upper, i)

            cv2.createTrackbar("{0}_{1}".format(id, key), "frame", min, max, change)

    createRGBTrackbar(False)
    createRGBTrackbar(True)


def create_hsv_bar():
    def trigger_changes(upper, i, x):
        target = LINE_COLOR_RANGE_HIGHER if upper else LINE_COLOR_RANGE_LOWER

        x = x if i == 0 else int(255 * x / 100.0)

        for index in range(3):
            if index == i:
                target[index] = x
            else:
                target[index] = target[index]

        render(params['current_frame'])

    def createHSVTrackbar(upper: bool):
        for i, key in enumerate(['h', 's', 'v']):
            min = (LINE_COLOR_RANGE_HIGHER if upper else LINE_COLOR_RANGE_LOWER)[i]
            max = 180 if key == 'h' else 100

            id = "upper" if upper else "lower"

            change = functools.partial(trigger_changes, upper, i)

            cv2.createTrackbar("{0}_{1}".format(id, key), "frame", min, max, change)

    createHSVTrackbar(False)
    createHSVTrackbar(True)


DATA = {
    "dir": "/Users/ozz/Desktop/",
    "targets": [
        {
            "name": "Screen Shot 2019-05-15 at 1.23.41 PM.png",
            "turn": "left"
        },
        {
            "name": "Screen Shot 2019-05-15 at 1.23.50 PM.png",
            "turn": "straight"
        },
        {
            "name": "Screen Shot 2019-05-15 at 4.32.18 PM.png",
            "turn": "right"
        },
        {
            "name": "Screen Shot 2019-05-15 at 5.24.29 PM.png",
            "turn": "right"
        },
        {
            "name": "Screen Shot 2019-05-15 at 4.31.38 PM.png",
            "turn": "right"
        },
        {
            "name": "Screen Shot 2019-05-15 at 4.31.38 PM.png",
            "turn": "right"
        },
        {
            "name": "Screen Shot 2019-05-15 at 5.23.47 PM.png",
            "turn": "straight"
        },
        {
            "name": "Screen Shot 2019-05-15 at 5.24.50 PM.png",
            "turn": "straight"
        },
        {
            "name": "Screen Shot 2019-05-16 at 8.21.49 PM.png",
            "turn": "left"
        },
        {
            "name": "Screen Shot 2019-05-16 at 8.21.56 PM.png",
            "turn": "left"
        },
        {
            "name": "Screen Shot 2019-05-16 at 8.22.06 PM.png",
            "turn": "left"
        },
    ]
}

if __name__ == "__main__":
    if configs.PRODUCTION_MODE or configs.LOCAL_MODE:
        main()
    else:
        params['kernel_max_width'] = size[0]
        params['kernel_width'] = int(params['kernel_max_width'] / 40)

        frame = cv2.imread(configs.single_image_source, cv2.IMREAD_COLOR)
        render(normalize_frame_for_lane_detection(frame))

    # @Data Unit Tests
    #

    # for target in DATA["targets"][-5:]:
    #     # reset
    #     direction["turn"] = None
    #     direction["angel"] = None
    #
    #     file = DATA["dir"] + target["name"]
    #
    #     frame = cv2.imread(file, cv2.IMREAD_COLOR)
    #     render(normalize_frame_for_lane_detection(frame))
    #     print(direction)
    #
    #     # if target["turn"] != direction["turn"]:
    #     #     print("doesn't match {}, expected={}, got={}".format(
    #     #         file, target["turn"], direction["turn"]
    #     #     ))
    #     #
    #     #     raise Exception()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
