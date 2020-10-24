import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import time


def transform(image, reverse=False):
    height = image.shape[0]
    width = image.shape[1]
    if not reverse:
        pts1 = np.float32([[330, 720], [448, 504], [1010, 720], [790, 504]])
        pts2 = np.float32([[330, 720], [330, 504], [1010, 720], [1010, 504]])
    else:
        pts2 = np.float32([[330, 720], [448, 504], [1010, 720], [790, 504]])
        pts1 = np.float32([[330, 720], [330, 504], [1010, 720], [1010, 504]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    image = cv2.warpPerspective(image, M, (width, height))
    if not reverse:
        cv2.imshow('transformed', image)
    else:
        cv2.imshow('reverse transformed', image)
    return image


def roi(image):
    # print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    rectangle = np.array(
        [[(200, height), (1000, height), (700, int(height/2)), (500, int(height/2))]])
    # print(triangle)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, rectangle, 255)
    # cv2.imshow('mask', mask)
    masked_image = cv2.bitwise_and(image, mask)
    # cv2.imshow('masked_image', masked_image)
    return masked_image


def detect_edge(image):
    img = np.copy(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.GaussianBlur(gray, (5, 5), 0)

    sharp = cv2.addWeighted(gray, 1, kernel, 0, 0)

    low_thres = 20
    high_thres = 50
    # blur = cv2.GaussianBlur(gray, (9, 9), 0)
    canny = cv2.Canny(sharp, low_thres, high_thres)
    cv2.imshow('edge', canny)
    return canny


def detect_line(image):
    # cv2.imshow('edge', image)
    lines = cv2.HoughLinesP(image, 1, np.pi/180, 20,
                            np.array([]), minLineLength=40, maxLineGap=20)

    if lines is None:
        print('No lines found!')
        return None

    # left_line_fit = []
    # right_line_fit = []
    # for line in lines:
    #     x1, y1, x2, y2 = line.squeeze()
    #     slope, intercept = np.polyfit((x1, x2), (y1, y2), deg=1)
    #     if slope < 0:
    #         left_line_fit.append((slope, intercept))
    #     else:
    #         right_line_fit.append((slope, intercept))

    # left_line_avg = np.mean(left_line_fit, axis=0) if len(
    #     left_line_fit) else None
    # right_line_avg = np.mean(
    #     right_line_fit, axis=0) if len(right_line_fit) else None

    left_line = []
    right_line = []
    for line in lines:
        x1, y1, x2, y2 = line.squeeze()
        if x1 < 640:
            left_line.append(line.squeeze())
        else:
            right_line.append(line.squeeze())

    left_line = np.mean(left_line, axis=0) if len(
        left_line) else None
    right_line = np.mean(right_line, axis=0) if len(right_line) else None

    lines = []
    if left_line is not None:
        lines.append(left_line)
    if right_line is not None:
        lines.append(right_line)

    # left_line = make_line(image, left_line_avg)
    # right_line = make_line(image, right_line_avg)
    return np.array(lines, dtype=np.int64)


def make_line(image, line):
    if line is None:
        return None
    y1 = image.shape[0]
    y2 = int(y1 * 0.4)
    x1 = int((y1 - line[1]) / line[0])
    x2 = int((y2 - line[1]) / line[0])
    return np.array([x1, y1, x2, y2])


def draw_line_with_frame(frame, lines):
    if lines is None:
        return frame
    line_img = np.zeros_like(frame)
    for line in lines:
        if line is not None:
            # print(line)
            x1, y1, x2, y2 = line
            cv2.line(line_img, (x1, y1), (x2, y2),
                     color=[255, 0, 0], thickness=10)
    line_img = transform(line_img, reverse=True)
    frame = cv2.addWeighted(frame, 0.8, line_img, 1, 0)
    return frame


def play_video(file):
    video = cv2.VideoCapture(file)
    while(video.isOpened()):
        _, frame = video.read()
        if frame is None:
            break
        img = np.copy(frame)
        img = transform(img)
        # img = roi(img)
        img = detect_edge(img)
        # img = roi(img)
        lines = detect_line(img)
        # print(lines)
        img = draw_line_with_frame(frame, lines)
        # if cv2.waitKey(0) == ord('c'):
        cv2.imshow('Press Q to quit!', img)
        # time.sleep(0.1)
        if cv2.waitKey(1) == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    path = os.path.curdir
    file = path + '/test2.mp4'
    play_video(file)
