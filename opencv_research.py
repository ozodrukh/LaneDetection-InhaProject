import cv2
import numpy as np
import functools, route, drivers
from datetime import datetime

# sudo modprobe bcm2835-v4l2 # to enable camera on pi

img_path = "/Users/ozz/Documents/Projects/opencv-py/data/outcpp.avi"
mask_image = cv2.imread("/Users/ozz/Documents/Projects/opencv-py/data/mask_path", cv2.IMREAD_GRAYSCALE)

size = (300, 300)

# camera.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
# camera.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

params = {
    'fps_counter': True,
    'morph': 'gradient',
    'kernel_width': 0,
    'kernel_max_width': 0,
    'kernel_height': 1,
    'current_frame': None,
}

original_hough_params = {
    'hough.rho': 1,
    'hough.theta': np.pi / 180,
    'hough.threshold': 100,
    'hough.min_distance': 100,
    'hough.max_line_gap': 50
}

modified_hough_params = {
    'hough.rho': 1,
    'hough.theta': np.pi / 180,
    'hough.threshold': 75,
    'hough.min_distance': 64,
    'hough.max_line_gap': 60
}

params.update(modified_hough_params)

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

    cv2.imshow("frame", original)
    # cv2.imshow("morphed", morphed)
    return original


def find_lines_using_contours(frame, morphed):
    #if cv2.getVersionMajor() >= 4:
    #    contours, _ = cv2.findContours(morphed, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_TREE)
    #else:
    _, contours, _ = cv2.findContours(morphed, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_TREE)

    if contours is None:
        print("no contours")

        cv2.putText(frame,
                    "No Contours",
                    (int(size[0] / 2), 0),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (255, 255, 255))
        return

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

    t, a = route.find_lines_on_contours(frame, contours, lambda rect: rect[1][0] > 10)

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
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Video not opened")
        return

    cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow("morphed", cv2.WINDOW_AUTOSIZE)

    cv2.moveWindow("frame", 0, 0)
    # cv2.moveWindow("morphed", 400, 0)

    attach_options_bar()
    create_hough_line_editor()

    if params["fps_counter"]:
        pass

    last_start = datetime.now()

    fps = 0
    current_fps = 0

    while camera.isOpened():
        ret, frame = camera.read()

        if frame.size == 0:
            break

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
        else:
            fps = fps + 1

            cv2.putText(frame, "fps: {}".format(current_fps),
                        (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

            cv2.imshow("frame", frame)
        try:
            key = cv2.waitKey(1)

            if key & 0xFF == ord('q'):
                break

            if key == 32:
                if cv2.waitKey(0) == 32:
                    continue
        except KeyboardInterrupt:
            pass

    camera.release()


def send_motor_signal():
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
        }
    ]
}

if __name__ == "__main__":
    main()

    # @Data Unit Tests
    #
    # params['kernel_max_width'] = size[0]
    # params['kernel_width'] = int(params['kernel_max_width'] / 40)
    #
    # # render(cv2.imread("/Users/ozz/Desktop/Screen Shot 2019-05-15 at 1.23.41 PM.png", cv2.IMREAD_COLOR))
    #
    # for target in DATA["targets"]:
    #     # reset
    #     direction["turn"] = None
    #     direction["angel"] = None
    #
    #     file = DATA["dir"] + target["name"]
    #
    #     render(cv2.imread(file, cv2.IMREAD_COLOR))
    #
    #     if target["turn"] != direction["turn"]:
    #         print("doesn't match {}, expected={}, got={}".format(
    #             file, target["turn"], direction["turn"]
    #         ))
    #
    #         raise Exception()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
