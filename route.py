from math import sqrt, pow

import cv2


def distance(pt1, pt2):
    return sqrt(pow(pt2[0] - pt1[0], 2) + pow(pt2[1] - pt1[1], 2))


def get_line_distortion(cnt, angel):
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])

    if angel >= 0:
        return rightmost
    else:
        return leftmost


def get_bottom_line(cnt):
    return tuple(cnt[cnt[:, :, 1].argmax()][0])


def intify(pt):
    return int(pt[0]), int(pt[1])


def find_lanes(frame, contours, filter_rect):
    frame_size = (frame.shape[1], frame.shape[0])
    center_point = intify((frame_size[0] / 2, frame_size[1]))

    cv2.line(frame, center_point, (center_point[0], 0), (80, 80, 80), 2)

    for i in range(len(contours)):
        rect = cv2.minAreaRect(contours[i])

        if not filter_rect(rect):
            continue

        rect_point = intify(get_bottom_line(contours[i]))

        d = distance(center_point, rect_point)

        if int(rect[2]) == 0 and rect[1][0] < 5:
            continue

        box = cv2.boxPoints(rect)

        for i in range(4):
            x, y = box[i]
            x1, y1 = box[(i + 1) % 4]

            cv2.line(frame, (x, y), (x1, y1), (0, 255, 0), 1)

        # dist_point = get_line_distortion(contours[i], rect[2])

        cv2.line(frame, center_point, intify(rect[0]), (0, 255, 0), 1)

        if rect[2] > 0:
            turn = "left"
        elif rect[2] > -40:
            turn = "right"
        else:
            turn = "straight"

        if frame_size[0] > rect_point[0] > 100:
            text_point = (center_point[0] - 50, rect_point[1])
        else:
            text_point = rect_point

        cv2.putText(frame,
                    "t: {3}, d: {0}, w: {1}, a={2}".format(int(d), int(rect[1][0]), int(rect[2]), turn),
                    text_point,
                    cv2.FONT_HERSHEY_PLAIN,
                    0.6,
                    (255, 255, 255))
