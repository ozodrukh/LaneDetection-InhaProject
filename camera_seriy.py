import numpy as np
import cv2
import math


class Car:

    def __init__(self, input_img):
        self.input_img = input_img                          #video frame

        self.img_center = 0
        self.line_center = 0

        self.left_flag = False                              #flag for left lanes
        self.right_flag = False                             #flag for right lanes

        self.selected_lines = np.empty((0, 4), dtype=int)   #lines where slope > 0.3
        self.sorted_lines = []                              #either left or right lanes

        self.left_pts = np.empty((0, 2), dtype=int)         #left lane points
        self.right_pts = np.empty((0, 2), dtype=int)        #right lane points

        self.init_y = 0
        self.final_y = 0
        # self.right_init_x
        # self.right_final_x
        self.right_m = 0.00
        self.right_b = np.array([], dtype=int)
        # self.left_init_x
        # self.left_final_x
        self.left_m = 0.00
        self.left_b = np.array([], dtype=int)
        self.regressed_lanes = np.zeros((4, 2), dtype=int)

        self.rows, self.cols = input_img.shape[:2]

    @staticmethod
    def remove_noise(input_img, kernel):
        output = cv2.GaussianBlur(input_img, (kernel, kernel), 0)
        return output

    @staticmethod
    def gray_scale(input_img):
        output = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
        return output

    @staticmethod
    def detect_edges(input_img, low_threshold=50, high_threshold=150):
        output = cv2.Canny(input_img, low_threshold, high_threshold)
        return output

    @staticmethod
    def detect_lines(input_img, min_line_len=20, max_line_gap=30):
        output = cv2.HoughLinesP(input_img, rho=1, theta=np.pi / 180,
                                 threshold=20, minLineLength=min_line_len, maxLineGap=max_line_gap)
        return output

    def line_separation(self, capture=0, lanes=0):
        slopes = []
        right_lines = np.empty((0, 4), dtype=int)  # flag for left lanes
        left_lines = np.empty((0, 4), dtype=int)  # flag for left lanes
        # x1 -> lane[0]
        # y1 -> lane[1]
        # x2 -> lane[2]
        # y2 -> lane[3]
        if lanes is not None:                       #select only lanes where slope > 0.3
            for i in lanes:
                slope = (i[0][3] - i[0][1]) / (i[0][2] - i[0][0] + 0.00001)
                if abs(slope) > 0.3:
                    slopes.append(slope)
                    self.selected_lines = np.append(self.selected_lines, i, axis=0)
            self.img_center = self.cols / 2

            # if capture is not None:
            #     self.img_center = 250
            # else:
            #     self.img_center = 500
            j = 0

            while j < len(self.selected_lines):         #from these selected lanes sort by left&right
                if slopes[j] > 0 and self.selected_lines[j][2] > self.img_center and \
                        self.selected_lines[j][0] > self.img_center:
                    right_lines = np.append(right_lines, [self.selected_lines[j]], axis=0)
                    self.right_flag = True
                elif slopes[j] < 0 and self.selected_lines[j][2] < self.img_center and \
                        self.selected_lines[j][0] < self.img_center:
                    left_lines = np.append(left_lines, [self.selected_lines[j]], axis=0)
                    self.left_flag = True
                j = j + 1

        self.sorted_lines.append(left_lines)            #append left&right lanes to one list
        self.sorted_lines.append(right_lines)           #sorted_lines[0] -> left, sorted_lines[1] -> right lines

    def regression(self, input_img):
        # rows, cols = input_img.shape[:2]
        # self.init_y = cols / 15
        # self.final_y = cols

        self.init_y = self.rows
        self.final_y = 70
        # print(self.init_y)

        if self.right_flag is True:
            for i in self.sorted_lines[1]:
                self.right_pts = np.append(self.right_pts, [[i[0], i[1]]], axis=0)
                self.right_pts = np.append(self.right_pts, [[i[2], i[3]]], axis=0)

            if len(self.right_pts) > 0:
                right_regression = cv2.fitLine(self.right_pts, cv2.DIST_L2, 0, 0.01, 0.01)
                self.right_m = right_regression[1][0] / right_regression[0][0]
                self.right_b = np.array([right_regression[2][0], right_regression[3][0]], dtype=int)
                right_init_x = ((self.init_y - self.right_b[1]) / self.right_m) + self.right_b[0]
                right_final_x = ((self.final_y - self.right_b[1]) / self.right_m) + self.right_b[0]
                self.regressed_lanes[0] = np.array([right_init_x, self.init_y], dtype=int)
                self.regressed_lanes[1] = np.array([right_final_x, self.final_y], dtype=int)

        if self.left_flag is True:
            for j in self.sorted_lines[0]:
                self.left_pts = np.append(self.left_pts, [[j[0], j[1]]], axis=0)
                self.left_pts = np.append(self.left_pts, [[j[2], j[3]]], axis=0)
            if len(self.left_pts) > 0:
                left_regression = cv2.fitLine(self.left_pts, cv2.DIST_L2, 0, 0.01, 0.01)
                self.left_m = left_regression[1][0] / left_regression[0][0]
                self.left_b = np.array([left_regression[2][0], left_regression[3][0]], dtype=int)
                left_init_x = ((self.init_y - self.left_b[1]) / self.left_m) + self.left_b[0]
                left_final_x = ((self.final_y - self.left_b[1]) / self.left_m) + self.left_b[0]
                self.regressed_lanes[2] = np.array([left_init_x, self.init_y], dtype=int)
                self.regressed_lanes[3] = np.array([left_final_x, self.final_y], dtype=int)

        return self.regressed_lanes

    def turn(self):
        output = " "
        left_sum = 0
        right_sum = 0

        # for i in self.left_pts:
        #     left_sum = left_sum + i[0]
        #
        # for j in self.right_pts:
        #     right_sum = right_sum + j[0]
        #
        # if len(self.left_pts > 0):
        #     left_sum = left_sum / len(self.left_pts)
        # if len(self.right_pts > 0):
        #     right_sum = right_sum / len(self.right_pts)
        # value = right_sum - left_sum
        # print("Value", value)

        if len(self.left_b) > 0 and len(self.right_b) > 0:
            x = ((self.right_m * self.right_b[0]) - (self.left_m * self.left_b[0]) - self.right_b[1] - self.left_b[1]) / (self.right_m - self.left_m)
            # print("X", x)
            # print("X: " + str(x) + " Img_Center: " + str(self.img_center))
            if x < self.img_center - 30:
                output = "Left"
            elif x > self.img_center + 30:
                output = "Right"
            elif x >= self.img_center + 25 or x <= self.img_center - 25:
                output = "Straight"
            # theta = theta + math.atan2((self.left_pts[0][0] - self.left_pts[0][0]), ())
            # if x < value - 30:
            #     output = "Left"
            # elif x > value + 30:
            #     output = "Right"
            # elif x >= value + 25 or x <= value - 25:
            #     output = "Straight"
        return output


def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv2.fillConvexPoly(mask, vertices, 255)
    else:
        cv2.fillConvexPoly(mask, vertices, (255,) * mask.shape[2])  # in case, the input image has a channel dimension
    return cv2.bitwise_or(image, mask)


def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    # rows, cols = image.shape[:2]
    # bottom_left = [cols * 0, rows * 1]
    # top_left = [cols * 0, rows * 0.7]
    # bottom_right = [cols * 1, rows * 1]
    # top_right = [cols * 1, rows * 0.7]
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)


# def regression(sorted_lanes):
# main function
# Capture video from file
cap = cv2.VideoCapture("/Users/ozz/Documents/Projects/opencv-py/data/outcpp.avi")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
width = cap.get(3)
height = cap.get(4)
fps = cap.get(cv2.CAP_PROP_FPS)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frameCount / fps
minutes = int(duration / 60)
seconds = duration % 60
# right_m = 0.0
# right_b = np.zeros([2, 1], dtype=int)
# left_m = 0.0
# left_b = np.zeros([2, 1], dtype=int)
counter = 0

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 640))
    cut = frame[320:639, 0:639]

    if ret is True:
        theta = 0
        # roi = select_region(frame)
        car = Car(cut)
        blurred = car.remove_noise(cut, kernel=1)
        gray = car.gray_scale(blurred)
        edges = car.detect_edges(gray)
        lanes = car.detect_lines(edges)
        car.line_separation(cap, lanes)
        regression_lane = car.regression(cut)
        cv2.line(cut, (regression_lane[0][0], regression_lane[0][1]),
                 (regression_lane[1][0], regression_lane[1][1]), (255, 0, 255), 5)
        cv2.line(cut, (regression_lane[2][0], regression_lane[2][1]),
                 (regression_lane[3][0], regression_lane[3][1]), (255, 0, 255), 5)
        # prediction = car.turn()
        # # print(prediction)
        # # q, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # # thresh = cv2.bitwise_not(thresh)
        left_theta = theta + math.atan2((regression_lane[1][1] - regression_lane[0][1]),
                                   (regression_lane[1][0] - regression_lane[0][0]))
        print("Left Theta:", theta)
        theta = 0
        right_theta = theta + math.atan2((regression_lane[3][1] - regression_lane[2][1]),
                                   (regression_lane[3][0] - regression_lane[2][0]))
        print("Right Theta:", theta)
        # print("Reg Lane:", regression_lane[1][0])
        if left_theta != 0 and right_theta == 0:
            cv2.putText(cut, "Left", (40, 40), 1, 3.0, (0, 255, 0), 1)
        elif left_theta == 0 and right_theta != 0:
            cv2.putText(cut, "Right", (40, 40), 1, 3.0, (0, 255, 0), 1)
        else:
            cv2.putText(cut, "Straight", (40, 40), 1, 3.0, (0, 255, 0), 1)

        cv2.imshow('Without Mask', cut)
        # cv2.imshow('With Mask', thresh_edges)
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()

# print('height: ' + str(height) + ' width: ' + str(width))
# print('fps = ' + str(fps))
# print('number of frames = ' + str(frameCount))
# print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))
# print(car.sorted_lines[0].shape)
# print(car.sorted_lines[1].shape)

#
# # Test on image
# left_pts = np.empty((0, 1, 2), dtype=int)
# right_pts = np.empty((0, 1, 2), dtype=int)
# regression_lane = np.zeros((4, 2), dtype=int)
# img = cv2.imread("/home/mikeygresl/Desktop/Lanes_2.jpg", 1)
# cut = img[380:639, 320:800]
# polygon = select_region(img)
# car = Car(polygon)
# blurred = car.remove_noise(polygon, kernel=5)
# gray = car.gray_scale(blurred)
# edges = car.detect_edges(gray)
# lanes = car.detect_lines(edges)
# car.line_separation(capture=polygon, lanes=lanes)
# regression_lane = car.regression(polygon)
# # q, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# # thresh = cv2.bitwise_not(thresh)
# cv2.line(polygon, (regression_lane[0][0], regression_lane[0][1]),
#          (regression_lane[1][0], regression_lane[1][1]), (255, 0, 255), 5)
# cv2.line(polygon, (regression_lane[2][0], regression_lane[2][1]),
#          (regression_lane[3][0], regression_lane[3][1]), (255, 0, 255), 5)
#
#         # cv2.line(img, (regression_lane[0][0], regression_lane[0][1]),
#         #          (regression_lane[1][0], regression_lane[1][1]), (0, 0, 255), 5)
#         # cv2.line(img, (regression_lane[2][0], regression_lane[2][1]),
#         #          (regression_lane[3][0], regression_lane[3][1]), (0, 0, 255), 5)
#
# cv2.imshow('lines', polygon)
# # cv2.imshow('edges', edges)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
