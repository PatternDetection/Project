import os
import glob
import pickle
from functools import wraps
from concurrent import futures

import cv2
import numpy as np
from PIL import Image
import yaml
from matplotlib import pyplot as plt
import layoutparser as lp
from tqdm import tqdm


def detect_wrapper(fn):
    @wraps(fn)
    def wrap(parser, im, *args, **kwargs):
        try:
            impath = None
            if isinstance(im, str):
                impath = im
                im = cv2.imread(im)
                im = im[:, :, ::-1]
            elif isinstance(im, Image.Image):
                im = np.ndarray(im)

            return fn(parser, im, *args, **kwargs)

        except Exception as e:
            print(f"Error: {e} {impath}")

    return wrap


def percent_to_px(im_h, im_w, bbx):
    left, top, right, bottom = bbx[:4]
    if right < 1 or bottom < 1:
        left = int(left * im_w)
        top = int(top * im_h)
        right = int(right * im_w)
        bottom = int(bottom * im_h)

    return left, top, right, bottom


def show_bbxes_on(im, bbxes, show=False, color=128):
    if isinstance(im, str):
        im = cv2.imread(im)

    h, w = im.shape[:2]
    for i, item in enumerate(bbxes):
        try:
            left, top, right, bottom, rtype, score = item
        except:
            left, top, right, bottom = item[:4]

        # Convert % to px
        left, top, right, bottom = percent_to_px(h, w, (left, top, right, bottom))
        cv2.rectangle(im, (left, top), (right, bottom), color=color, thickness=5)

    if show:
        plt.imshow(im)
        plt.show()


class LayoutBaseParser(object):

    def detect(self, im, *args, **kwargs):
        # This function returns the layout of the query image in an internal format
        # Call .dump() to get a general type to save
        raise NotImplementedError

    def draw(self, im, layout, *args, **kwargs):
        raise NotImplementedError

    def dump(self, im, layout):
        # This function dumps the detected layout to an array:
        # left(%), top(%), right(%), bottom(%), bbx_type(str)
        raise NotImplementedError

    def batch_detect(self, img_folder, start=-1, end=-1):
        def fn(impath):
            x = cv2.imread(impath)
            r = self.detect(impath)

            return {
                "path": impath,
                "h": x.shape[0],
                "w": x.shape[1],
                "layout": r
            }

        impaths = []

        for dirpath, dirnames, filenames in os.walk(img_folder):
            valids = filter(lambda x: x.endswith(".jpg"), filenames)
            files = map(lambda f: os.path.join(dirpath, f), valids)
            impaths.extend(files)

        impaths = list(sorted(impaths))
        if end > start >= 0:
            impaths = impaths[start: end]
        print(f"Process {len(impaths)} images.")

        with futures.ThreadPoolExecutor() as executor:
            # results = list(tqdm(executor.map(self.detect, impaths), total=len(impaths)))
            results = list(tqdm(executor.map(fn, impaths), total=len(impaths)))

        # if not img_folder.endswith("/"):
        #     img_folder += "/"
        # return list(zip(map(lambda x: x.replace(img_folder, ""), impaths), results))
        # return list(zip(impaths, results))
        return results


class HarvardLayoutParser(LayoutBaseParser):
    # Source code: https://github.com/Layout-Parser/layout-parser
    # Model zoo: https://layout-parser.readthedocs.io/en/latest/notes/modelzoo.html

    PresetLabels = {
        "HJDataset": {1: "Page Frame", 2: "Row", 3: "Title Region", 4: "Text Region", 5: "Title", 6: "Subtitle",
                      7: "Other"},
        "PubLayNet": {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        "PrimaLayout": {1: "TextRegion", 2: "ImageRegion", 3: "TableRegion", 4: "MathsRegion", 5: "SeparatorRegion",
                        6: "OtherRegion"},
        "NewspaperNavigator": {0: "Photograph", 1: "Illustration", 2: "Map", 3: "Comics/Cartoon",
                               4: "Editorial Cartoon",
                               5: "Headline", 6: "Advertisement"}

    }

    def __init__(self, model_name, model_path=None, config_path=None, label_map=None, score_thresh=0.5):
        if label_map is None:
            label_map = HarvardLayoutParser.PresetLabels.get(model_name, None)

        self.model = lp.Detectron2LayoutModel(config_path,
                                              model_path=model_path,
                                              extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", score_thresh],
                                              label_map=label_map)

    @staticmethod
    def is_text(type_cls):
        try:
            s = type_cls.lower()
            return "text" in s or "title" in s or "headline" in s
        except:
            return True

    @detect_wrapper
    def detect(self, im, keep_text_only=True, dump_to_tuples=True, **kwargs):
        layout = self.model.detect(im)
        if keep_text_only:
            layout = lp.Layout([b for b in layout if self.is_text(b.type)])

        if dump_to_tuples:
            return self.dump(im, layout)

        return layout

    def draw(self, im, layout, **kwargs):
        return lp.draw_box(im, layout, box_width=5, show_element_id=True)

    def dump(self, im, layout):
        bbxes = []

        h, w = im.shape[:2]
        for t in layout:
            x1, y1, x2, y2 = t.coordinates
            x1 /= w
            x2 /= w
            y1 /= h
            y2 /= h

            bbxes.append((x1, y1, x2, y2, t.type, t.score))

        # Sort by score
        bbxes.sort(key=lambda x: x[-1], reverse=True)
        return bbxes


if __name__ == '__main__':
    models_dir = "../../models"
    jpgs_dir = "../../data/jpgs"
    layout_output_dir = "../../data/layout"
    vis_output_dir = "../../data/vis"
    score_thresh = 0.2
    override = False
    vis = True

    for dataset in os.listdir(models_dir):
        subdir = os.path.join(models_dir, dataset)
        configs = glob.glob(os.path.join(subdir, "*.yaml"))

        for conf in configs:
            pth = conf.replace("yaml", "pth")
            name = f"{dataset}-{os.path.basename(conf).split('.')[0]}"
            print(f"===={name}====")

            parser = HarvardLayoutParser(dataset,
                                         model_path=pth,
                                         config_path=conf,
                                         score_thresh=score_thresh)

            layout_outpath = os.path.join(layout_output_dir, f"{name}")
            if override or not os.path.exists(layout_outpath):
                results = parser.batch_detect(jpgs_dir)
                print(results[0])
                pickle.dump(results, open(layout_outpath, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            else:
                results = None

            if vis:
                if results is None:
                    results = pickle.load(open(layout_outpath, "rb"))

                this_dir = os.path.join(vis_output_dir, name)
                if os.path.exists(this_dir):
                    continue

                os.makedirs(this_dir, exist_ok=False)


                def draw_and_save(item):
                    impath = item["path"]
                    layout = item["layout"]
                    x = cv2.imread(impath)
                    show_bbxes_on(x, layout)
                    cv2.imwrite(os.path.join(this_dir, os.path.basename(impath)), x)


                with futures.ThreadPoolExecutor() as executor:
                    results = list(tqdm(executor.map(draw_and_save, results), total=len(results)))
