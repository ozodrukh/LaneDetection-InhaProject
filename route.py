from math import sqrt, pow, atan, pi

import cv2


def distance(pt1, pt2):
    return sqrt(pow(pt2[0] - pt1[0], 2) + pow(pt2[1] - pt1[1], 2))


def get_bounds(cnt):
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    return [leftmost, topmost, rightmost, bottommost]


def intify(pt):
    return int(pt[0]), int(pt[1])


def find_lines_on_hough_lines(frame, lines):
    if lines is not None and len(lines) > 0:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)


def get_contour_point(contours, index, point_index):
    line = contours[index][point_index, 0]
    return line[0], line[1]


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])  # Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    point = x, y

    def lies_in(a, b, x):
        if a[0] >= b[0]:
            r1 = a[0] >= x[0] >= b[0]
        else:
            r1 = b[0] >= x[0] >= a[0]

        if a[1] >= b[1]:
            r2 = a[1] >= x[1] >= b[1]
        else:
            r2 = b[1] >= x[1] >= a[1]

        return r1 and r2

    if lies_in(min(line1), max(line1), point) and lies_in(min(line2), max(line2), point):
        return point
    else:
        raise Exception('lines do not intersect')


def angel(line):
    return angel2(line[0][0], line[0][1], line[1][0], line[1][1])


def angel2(x1, y1, x2, y2):
    return atan((y2 - y1) / (x2 - x1)) / pi * 180


def find_lines_on_contours(frame, contours, filter_rect):
    frame_size = (frame.shape[1], frame.shape[0])
    center_point = intify((frame_size[0] / 2, frame_size[1]))

    cv2.line(frame, center_point, (center_point[0], 0), (80, 80, 80), 2)

    contours_data = []

    for i in range(len(contours)):
        rect = cv2.minAreaRect(contours[i])

        if not filter_rect(rect):
            continue

        bounds = get_bounds(contours[i])
        rect_point = intify(bounds[3])

        d = distance(center_point, rect_point)

        if int(rect[2]) == 0 and rect[1][0] < 50:
            continue

        line_point = [bounds[3]]

        ignore_shift = 15

        left_distance = distance(line_point[0], bounds[0])
        right_distance = distance(line_point[0], bounds[2])

        if right_distance > left_distance > ignore_shift:
            min_distance_point = bounds[0]
        elif right_distance > ignore_shift:
            min_distance_point = bounds[2]
        else:
            min_distance_point = bounds[0]

        line_point.append(min_distance_point)

        cv2.line(frame, line_point[1], line_point[0], (255,255,255), 2)

        try:
            point = line_intersection([(center_point[0], 0), center_point], line_point)
            #print([(center_point[0], 0), center_point], line_point, point)
        except Exception:
            #print("intersection not found on ctr={}".format(i))
            continue

        #print("intersection found on ctr={}".format(i))
        cv2.circle(frame, intify(point), 5, (255, 255, 255), 1)

        # print(angel(line_point))

        contours_data.append({
            "index": i,
            "bounds": rect,
            "angel": angel([bounds[3], point]),
            "contour_line": contours[i],
            "center_point": center_point,
            "line_point": line_point,
            "cross_point": point,
            "width": rect[1][0],
            "distance": distance(center_point, bounds[3])
        })

        continue

    target = None

    for data in contours_data:
        if 15 > abs(data["angel"]):
            continue

        if target is None or target["distance"] > data["distance"]:
            target = data

        # print("index: {}, angel: {}, distance: {}, width: {}".format(
        #     data["index"],
        #     data["angel"],
        #     data["distance"],
        #     data["width"],
        # ))

    if target is None:
        print("no intersaction found, going straight")

        cv2.putText(frame,
                    "straight",
                    center_point,
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (255, 255, 255))
        return "straight", -70

    rect = target["bounds"]
    rect_point = intify(target["line_point"][1])

    box = cv2.boxPoints(rect)

    for i in range(4):
        x, y = box[i]
        x1, y1 = box[(i + 1) % 4]

        cv2.line(frame, (x, y), (x1, y1), (0, 255, 0), 1)

    cv2.line(frame, intify(center_point), intify(target["cross_point"]), (255, 255, 255), 5)

    cv2.line(frame,
             target["line_point"][0],
             intify(target["cross_point"]),
             (255, 255, 255),
             5)

    # dist_point = get_line_distortion(contours[i], rect[2])

    # cv2.line(frame, center_point, intify(rect[0]), (0, 255, 0), 1)

    if target["angel"] > 0:
        turn = "left"
    elif target["angel"] > -60:
        turn = "right"
    else:
        turn = "straight"

    #print("angel={}, turn={}".format(target["angel"], turn))

    if frame_size[0] > rect_point[0] > 100:
        text_point = (center_point[0] - 50, rect_point[1])
    else:
        text_point = rect_point

    cv2.putText(frame,
                "{}, {}".format(turn, target["angel"]),
                center_point,
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 255, 255))

    cv2.putText(frame,
                "t: {3}, d: {0}, w: {1}, a={2}".format(int(d), int(rect[1][0]), int(rect[2]), turn),
                text_point,
                cv2.FONT_HERSHEY_PLAIN,
                0.6,
                (255, 255, 255))

    return turn, target["angel"]
