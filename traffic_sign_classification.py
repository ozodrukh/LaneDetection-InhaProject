from configs import GUI_MODE

import sys
import cv2
import numpy as np

STRUCTURE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))

HSV_BLUE_LOWER = np.array([100, 50, 50])
HSV_BLUE_UPPER = np.array([124, 255, 255])

HSV_RED_LOWER = np.array([-10, 50, 50])
HSV_RED_UPPER = np.array([10, 255, 255])


def try_detect_sign(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_blue = cv2.inRange(hsv, HSV_BLUE_LOWER, HSV_BLUE_UPPER)
    mask_red = cv2.inRange(hsv, HSV_RED_LOWER, HSV_RED_UPPER)

    if GUI_MODE:
        cv2.imshow('frame', frame)
        cv2.imshow('color_space', hsv)
        cv2.imshow('blue_mask', mask_blue)
        cv2.imshow('red_mask', mask_red)

    res, i = filter_noise(mask_blue, frame, 0)
    res, i = filter_noise(mask_red, frame, i)
    return res


def filter_noise(mask, frame, i):
    target = cv2.blur(mask, (9, 9))

    _, target = cv2.threshold(target, 127, 255, cv2.THRESH_BINARY)
    closed = cv2.morphologyEx(target, cv2.MORPH_CLOSE, STRUCTURE_KERNEL)

    if GUI_MODE:
        cv2.imshow('target_threshold', target)
        cv2.imshow('target_closed', closed)

    erode = cv2.erode(closed, None, iterations=4)
    dilate = cv2.dilate(erode, None, iterations=10)

    return classify_image(dilate, frame, i)


def classify_image(dilate, frame, i):
    import os

    parent_dir = "/data/trained_data/"
    classifiers = []

    for file in os.listdir(parent_dir):
        classifiers.append(cv2.CascadeClassifier(file))

    contours, hierarchy = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    res = frame.copy()
    for con in contours:
        rect = cv2.minAreaRect(con)
        box = np.int0(cv2.boxPoints(rect))

        print([box])

        h1 = max([box][0][0][1], [box][0][1][1], [box][0][2][1], [box][0][3][1])
        h2 = min([box][0][0][1], [box][0][1][1], [box][0][2][1], [box][0][3][1])
        l1 = max([box][0][0][0], [box][0][1][0], [box][0][2][0], [box][0][3][0])
        l2 = min([box][0][0][0], [box][0][1][0], [box][0][2][0], [box][0][3][0])

        if h1 - h2 > 0 and l1 - l2 > 0:
            temp = frame[h2:h1, l2:l1]
            try:
                if GUI_MODE:
                    cv2.imshow(str(i), temp)

                gray = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)

                for target_classifier in classifiers:
                    i = has_similarities(target_classifier, gray, temp, i)

            except OSError as err:
                print("OS error: {0}".format(err))
            except ValueError:
                print("Could not convert data to an integer.")
            except:
                print("Unexpected error:", sys.exc_info()[0])
    return res, i


def with_mask(frame, mask):
    return cv2.bitwise_or(frame, mask)


def has_similarities(cascade, gray, frame, i=0):
    found = cascade.detectMultiScale(gray, 1.3, 5)

    if len(found) is not 0:
        for (x, y, w, h) in found:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_color = frame[y:y + h, x:x + w]

            i += 1

            if GUI_MODE:
                cv2.imshow('detected color ' + str(i), roi_color)

    return i
