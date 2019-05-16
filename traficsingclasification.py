import sys

import cv2
import numpy as np


def circle(dilate, img):
    circle = cv2.HoughCircles(dilate.copy(), cv2.HOUGH_GRADIENT, 1, 1)
    # print('轮廓个数：', len(circle))
    print(circle)
    i = 0
    res = img.copy()
    return res


def addmask(mask1, mask2):
    mask = cv2.bitwise_or(mask1, mask2)
    return mask


def processing(mask, img, i):
    blurred = cv2.blur(mask, (9, 9))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))

    # cv2.imshow('blurred', blurred)

    ret, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('blurred binary', binary)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('closed', closed)

    # 腐蚀和膨胀
    '''
    腐蚀操作将会腐蚀图像中白色像素，以此来消除小斑点，
    而膨胀操作将使剩余的白色像素扩张并重新增长回去。
    '''
    erode = cv2.erode(closed, None, iterations=4)
    # cv2.imshow('erode', erode)
    dilate = cv2.dilate(erode, None, iterations=10)
    # cv2.imshow('dilate', dilate)

    # 查找轮廓
    res, i = rectangle(dilate, img, i)
    # res = circle(dilate, res)
    # 显示画了标志的原图
    return res, i


def trafficsSingDetection(img):
    print(img, type(img), img.shape[:2], img.dtype)
    # cv2.imshow('img', img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imshow('hsv', hsv)

    blue_lower = np.array([100, 50, 50])
    blue_upper = np.array([124, 255, 255])
    red_lower = np.array([-10, 50, 50])
    red_upper = np.array([10, 255, 255])
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)
    mask_red = cv2.inRange(hsv, red_lower, red_upper)
    print('mask', type(mask_blue), mask_blue.shape)
    cv2.imshow('mask', mask_blue)
    res = img
    i = 0
    res, i = processing(mask_blue, res, i)

    res, i = processing(mask_red, img, i)

    # cv2.imshow('res', res)
    return res


def rectangle(dilate, img, i):
    cascade_circle = cv2.CascadeClassifier('cascade_circle_lgb.xml')
    cascade_left = cv2.CascadeClassifier('cascade_left_lgb.xml')
    cascade_parking = cv2.CascadeClassifier('cascade_parking_lgb.xml')
    cascade_right = cv2.CascadeClassifier('cascade_right.xml')
    cascade_stop = cv2.CascadeClassifier('cascade_stop_lgb.xml')
    cascade_stop_2=cv2.CascadeClassifier('cascade_stop.xml')
    cascade_pedestrian=cv2.CascadeClassifier('pedestrian.xml')
    cascades = (cascade_right, cascade_parking, cascade_left, cascade_circle, cascade_stop,cascade_stop_2,cascade_pedestrian)
    contours, hierarchy = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print('轮廓个数：', len(contours))

    res = img.copy()
    for con in contours:
        # 轮廓转换为矩形
        rect = cv2.minAreaRect(con)
        # 矩形转换为box
        box = np.int0(cv2.boxPoints(rect))
        # 在原图画出目标区域
        #draw target region
        # cv2.drawContours(res, [box], -1, (0, 255, 255), 2)
        print([box])
        # 计算矩形的行列
        h1 = max([box][0][0][1], [box][0][1][1], [box][0][2][1], [box][0][3][1])
        h2 = min([box][0][0][1], [box][0][1][1], [box][0][2][1], [box][0][3][1])
        l1 = max([box][0][0][0], [box][0][1][0], [box][0][2][0], [box][0][3][0])
        l2 = min([box][0][0][0], [box][0][1][0], [box][0][2][0], [box][0][3][0])
        print('h1', h1)
        print('h2', h2)
        print('l1', l1)
        print('l2', l2)
        # 加上防错处理，确保裁剪区域无异常
        if h1 - h2 > 0 and l1 - l2 > 0:
            # 裁剪矩形区域
            temp = img[h2:h1, l2:l1]
            try:
                cv2.imshow(str(i), temp)
                gray = cv2.cvtColor(temp, cv2.COLOR_RGB2GRAY)
                for cascade in cascades:
                    i = detectation(cascade, gray, temp, i)
            except OSError as err:
                print("OS error: {0}".format(err))
            except ValueError:
                print("Could not convert data to an integer.")
            except:
                print("Unexpected error:", sys.exc_info()[0])

            # i = i + 1
            # 显示裁剪后的标志
            # cv2.imshow('sign' + str(i), temp)

    return res, i


def detectation(cascad, gray, img, i=0):
    found = cascad.detectMultiScale(gray, 1.3, 5)
    print('len=', len(found))
    if len(found) is not 0:
        for (x, y, w, h) in found:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            # cv2.imshow('detected gray ' + str(i), roi_gray)
            cv2.imshow('detected color ' + str(i), roi_color)
            i += 1
    return i


def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        img = cv2.imread('屏幕快照 2019-05-12 下午9.14.27.png')

        res = trafficsSingDetection(img)
        cv2.imshow('res', res)
        img = cv2.imread('01892_00002.ppm')
        cv2.imshow('pic', img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
