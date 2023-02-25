import cv2
import numpy as np
import os


def otsu_seg(img):
    ret_th, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return ret_th, bin_img


def find_pole(bin_img):
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = 0
    for i in range(len(contours)):
        area += cv2.contourArea(contours[i])
    area_mean = area / len(contours)
    mark = []
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) < area_mean:
            mark.append(i)
    return contours, hierarchy, mark


def draw_box(img, contours):
    img = cv2.rectangle(img,
            (contours[0][0], contours[0][1]),
            (contours[1][0], contours[1][1]),
            (255, 255, 255), 3)
    return img


def main(img):
    ret, th = otsu_seg(img)
    contours, hierarchy, mark = find_pole(th)
    upper_left = []
    low_right = []
    for i in range(len(contours)):
        if i not in mark:
            left_point = contours[i].min(axis=1).min(axis=0)
            right_point = contours[i].max(axis=1).max(axis=0)
            if abs(left_point[0] - right_point[0]) >= 460:
                continue
            upper_left.append(left_point)
            low_right.append(right_point)
    areas = []
    for x, y in zip(upper_left, low_right):
        area = abs(x[0] - y[0]) * abs(x[1] - y[1])
        areas.append(area)
    best = areas.index(max(areas))
    best_left, best_right = upper_left[best], low_right[best]
    img = draw_box(img, (best_left, best_right))
    return img


def DetImg(img):
    h = img.shape[0]
    w = img.shape[1]
    cropped = img[:, 355:816]
    cropped_l = img[:, 0:355]
    cropped_r = img[:, 816:w]
    # cv2.imshow('cropped', cropped)
    # cv2.waitKey(0)
    res_img = main(cropped)
    show_img = np.concatenate((cropped_l, res_img, cropped_r), axis=1)
    cv2.line(show_img, (355, h), (355, 0), (0, 0, 255), 2)
    cv2.line(show_img, (816, h), (816, 0), (0, 0, 255), 2)
    return show_img


if __name__ =="__main__":
    img = cv2.imread(r'imgs\6\6-4.jpg', cv2.IMREAD_GRAYSCALE)
    show_img = DetImg(img)
    cv2.imshow('results', show_img)
    cv2.waitKey(0)





