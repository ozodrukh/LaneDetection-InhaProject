import cv2
import numpy as np

DEFAULT_KERNEL = (3, 3)

red = 0
green = 65

red_lower = np.array([red - 10, 100, 100])
red_upper = np.array([red + 10, 255, 255])

green_lower = np.array([green - 10, 100, 100])
green_upper = np.array([green + 10, 255, 255])

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, DEFAULT_KERNEL)


def get_traffic_light_color(frame) -> str:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    red_res = cv2.bitwise_and(frame, frame, mask=red_mask)
    green_res = cv2.bitwise_and(frame, frame, mask=green_mask)

    red_closing = cv2.morphologyEx(red_res, cv2.MORPH_CLOSE, kernel)
    green_closing = cv2.morphologyEx(green_res, cv2.MORPH_CLOSE, kernel)

    red_gray = cv2.cvtColor(red_closing, cv2.COLOR_BGR2GRAY)
    green_gray = cv2.cvtColor(green_closing, cv2.COLOR_BGR2GRAY)

    (thresh1, red_bw) = cv2.threshold(red_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    (thresh2, green_bw) = cv2.threshold(green_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    red_black = cv2.countNonZero(red_bw)
    green_black = cv2.countNonZero(green_bw)

    print(red_black)
    print(green_black)

    if red_black > 500:
        return "red"
    elif green_black > 500:
        return "green"


if __name__ == "__main__":
    images = ["trafficLights.jpg", "trafficLightsRed.jpg", "traffic_light_green.jpg"]

    for data in images:
        frame = cv2.imread("data/" + data, cv2.IMREAD_COLOR)
        frame = cv2.resize(frame, (300, 300))
        print(get_traffic_light_color(frame))
