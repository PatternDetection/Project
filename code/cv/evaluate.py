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
    """ Use px to represent bbx """
    impath = item["path"]
    preds = item["layout"]
    h = item["h"]
    w = item["w"]

    bbxes = []

    for l, t, r, b, c, s in preds:
        if (r - l) < min_width or s < score_thresh:
            continue

        l = int(w * l)
        r = int(w * r)
        t = int(h * t)
        b = int(h * b)

        bbxes.append((l, t, r, b))

    return impath, bbxes


def count_intersection(pred, gt, offset=1):
    return np.clip(np.min([pred[:, 2], gt[:, 2]], axis=0) + offset - np.max([pred[:, 0], gt[:, 0]], axis=0), a_min=0,
                   a_max=None, ) * np.clip(
        np.min([pred[:, 3], gt[:, 3]], axis=0) + offset - np.max([pred[:, 1], gt[:, 1]], axis=0), a_min=0, a_max=None)


def count_area(coord, offset=1):
    l, t, r, b = coord
    return (r - l + offset) * (b - t + offset)


def cal_max_iox(pred, gt, if_ioa=False, offset=1):
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

    ## IoU
    # a1 = count_area(gt[k], offset)
    # a2 = count_area(pred[k], offset)
    #
    # return ai / (a1 + a2 - ai)

    ## IoA
    return ai / count_area(gt[k], offset)


def cal_accumulated_iox(pred, gt, if_ioa=False, offset=1, joint_thresh=0.2):
    if len(pred) == 0:
        return 0, None

    pred = np.array(pred).reshape(-1, 4)
    gt = np.array(gt).reshape(1, 4)
    gt = np.tile(gt, len(pred)).reshape(-1, 4)

    a1 = count_area(gt[0], offset)
    thresh = a1 * joint_thresh

    intersections = count_intersection(pred, gt, offset)
    k = intersections > thresh

    if np.sum(k) <= 0:
        return 0, None

    left = np.min(pred[k, 0])
    top = np.min(pred[k, 1])
    right = np.max(pred[k, 2])
    bottom = np.max(pred[k, 3])

    rect = [left, top, right, bottom]
    pred = np.array([rect])
    ai = count_intersection(pred, gt[:1], offset)
    a1 = count_area(gt[0], offset)

    if if_ioa:
        ioa = ai / a1
        return ioa.item(), rect

    a2 = count_area(pred[0], offset)
    return (ai / (a1 + a2 - ai)).item(), rect
