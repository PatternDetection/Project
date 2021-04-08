import os
import pickle
import glob
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def parse_label_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    bbxes = []

    for item in root.iter('object'):
        cls = item.find("name").text
        ymin = int(item.find("bndbox/ymin").text)
        xmin = int(item.find("bndbox/xmin").text)
        ymax = int(item.find("bndbox/ymax").text)
        xmax = int(item.find("bndbox/xmax").text)

        bbxes.append((cls, (xmin, ymin, xmax, ymax)))

    return os.path.basename(root.find('filename').text), bbxes


def parse_pred(item, min_width=0.5, score_thresh=0.5):
    imgname, preds = item

    t = cv2.imread(imgname)
    h, w = t.shape[:2]
    bbxes = []

    for l, t, r, b, c, s in preds:
        if (r - l) < min_width or s < score_thresh:
            continue

        l = int(w * l)
        r = int(w * r)
        t = int(h * t)
        b = int(h * b)

        bbxes.append((l, t, r, b))

    return os.path.basename(imgname), bbxes


def count_intersection(pred, gt, offset=1):
    return (np.min([pred[:, 2], gt[:, 2]], axis=0) + offset - np.max([pred[:, 0], gt[:, 0]], axis=0)) * (
            np.min([pred[:, 3], gt[:, 3]], axis=0) + offset - np.max([pred[:, 1], gt[:, 1]], axis=0))


def count_area(coord, offset=1):
    l, t, r, b = coord
    return (r - l + offset) * (b - t + offset)


def cal_max_iou(pred, gt, offset=1):
    if len(pred) == 0:
        return 0

    pred = np.array(pred).reshape(-1, 4)
    gt = np.array(gt).reshape(1, 4)
    gt = np.tile(gt, len(pred)).reshape(-1, 4)

    intersections = count_intersection(pred, gt, offset)

    k = np.argmax(intersections)
    ai = intersections[k]
    if ai <= 0:
        return 0

    a1 = count_area(gt[k])
    a2 = count_area(pred[k])

    return ai / (a1 + a2 - ai)


def cal_accumulated_iou(pred, gt, offset=1):
    if len(pred) == 0:
        return 0, None

    pred = np.array(pred).reshape(-1, 4)
    gt = np.array(gt).reshape(1, 4)
    gt = np.tile(gt, len(pred)).reshape(-1, 4)

    intersections = count_intersection(pred, gt, offset)
    k = intersections > 0

    if np.sum(k) <= 0:
        return 0, None

    left = np.min(pred[k, 0])
    top = np.min(pred[k, 1])
    right = np.max(pred[k, 2])
    bottom = np.max(pred[k, 3])

    pred = np.array([[left, top, right, bottom]])
    ai = count_intersection(pred, gt[:1], offset)
    a1 = count_area(gt[0])
    a2 = count_area(pred[0])

    return ai / (a1 + a2 - ai), (left, top, right, bottom)