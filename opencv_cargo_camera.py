import cv2
import numpy as np
import os
import time


def find_pole(img):
    ret_th, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
    box_h = abs(contours[1][1] - contours[0][1])
    box_w = abs(contours[1][0] - contours[0][0])
    img = cv2.line(img, (contours[0][0], contours[0][1] + int(box_h/2)),
                   (contours[1][0], contours[0][1] + int(box_h/2)), (255, 255, 255), 3)
    return img


def main(img):
    contours, hierarchy, mark = find_pole(img)
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
    try:
        for x, y in zip(upper_left, low_right):
            area = abs(x[0] - y[0]) * abs(x[1] - y[1])
            areas.append(area)
        best = areas.index(max(areas))
        best_left, best_right = upper_left[best], low_right[best]
        img = draw_box(img, (best_left, best_right))
    except:
        best_left, best_right = 0, 0
    return img, best_left, best_right


def DetImg(img):
    h = img.shape[0]
    w = img.shape[1]
    cropped = img[:, 355:816]
    cropped_l = img[:, 0:355]
    cropped_r = img[:, 816:w]
    # cv2.imshow('cropped', cropped)
    # cv2.waitKey(0)
    res_img, best_left, best_right = main(cropped)
    show_img = np.concatenate((cropped_l, res_img, cropped_r), axis=1)
    show_img_bgr = cv2.cvtColor(show_img, cv2.COLOR_GRAY2BGR)
    cv2.line(show_img_bgr, (355, h), (355, 0), (0, 0, 255), 2)
    cv2.line(show_img_bgr, (816, h), (816, 0), (0, 0, 255), 2)
    cv2.line(show_img_bgr, (0, int(h / 2)), (w, int(h / 2)), (0, 0, 255), 2)
    try:
        box_centre = int(best_left[1] + abs(best_left[1] - best_right[1]) / 2)
    except:
        box_centre = 0
    return show_img_bgr, box_centre, int(h/2)


if __name__ == "__main__":
    filePath = r'111_2020_05_14.mp4'
    cap = cv2.VideoCapture(filePath)
    sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', 'e', 'g')
    fourcc = cv2.VideoWriter_fourcc(*'mpeg')
    ## open and set props
    vout = cv2.VideoWriter()
    vout.open('output.mpeg', fourcc, 20, sz, True)
    prev_time = time.time()
    interval_time = 0
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            show_img, box_centre, img_centre = DetImg(frame)
            if box_centre != 0:
                if abs(box_centre - img_centre) <= 10:
                    current_time = time.time()
                    interval_time = abs(prev_time - current_time)
                    prev_time = current_time
                    print(interval_time)
            label = 'Time interval is: {}'.format(interval_time)
            cv2.putText(show_img, label, (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.imshow("video", show_img)
            vout.write(show_img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    vout.release()





