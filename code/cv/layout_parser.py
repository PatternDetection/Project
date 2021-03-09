import os
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

        except:
            print(f"Error: {impath}")

    return wrap


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

    def batch_detect(self, img_folder, start=0, end=100):

        impaths = []

        for dirpath, dirnames, filenames in os.walk(img_folder):
            valids = filter(lambda x: x.endswith(".jpg"), filenames)
            files = map(lambda f: os.path.join(dirpath, f), valids)
            impaths.extend(files)

        impaths = list(sorted(impaths))[start: end]
        print(f"Found {len(impaths)} images.")

        with futures.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(self.detect, impaths), total=len(impaths)))

        if not img_folder.endswith("/"):
            img_folder += "/"

        return list(zip(map(lambda x: x.replace(img_folder, ""), impaths), results))

    @staticmethod
    def show_bbxes_on(im, bbxes, show=False):
        h, w = im.shape[:2]        
        for i, (left, top, right, bottom, rtype, score) in enumerate(bbxes):
            # Convert % to px
            if right < 1 or bottom < 1:
                left = int(left * w)
                top = int(top * h)
                right = int(right * w)
                bottom = int(bottom * h)

            cv2.rectangle(im, (left, top), (right, bottom), color=128, thickness=5)

        if show:
            plt.imshow(im)
            plt.show()

    @staticmethod
    def verify_saved_dets(pkl_path, im_root, draw_dir):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        for relpath, result in data:
            savepath = os.path.join(draw_dir, relpath.replace("/", "-"))
            impath = os.path.join(im_root, relpath)

            im = cv2.imread(impath)
            LayoutBaseParser.show_bbxes(im, result)
            cv2.imwrite(savepath, im)


class HarvardLayoutParser(LayoutBaseParser):
    # Source code: https://github.com/Layout-Parser/layout-parser
    # Model zoo: https://layout-parser.readthedocs.io/en/latest/notes/modelzoo.html

    PresetLabels = {
        "HJDataset": {1: "Page Frame", 2: "Row", 3: "Title Region", 4: "Text Region", 5: "Title", 6: "Subtitle",
                      7: "Other"},
        "PubLayNet": {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        "Prima": {1: "TextRegion", 2: "ImageRegion", 3: "TableRegion", 4: "MathsRegion", 5: "SeparatorRegion",
                  6: "OtherRegion"},
        "Newspaper": {0: "Photograph", 1: "Illustration", 2: "Map", 3: "Comics/Cartoon", 4: "Editorial Cartoon",
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
        s = type_cls.lower()
        return "text" in s or "title" in s or "headline" in s

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
    with open("../config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    hlp_conf = config["LAYOUT"]["HarvardLP"]
    dataset = "Prima"
    modelname = hlp_conf["OpenModels"][dataset][0]
    score_thresh = 0.2
    parser = HarvardLayoutParser(dataset,
                                 model_path=os.path.join(config["ROOT"], hlp_conf["ModelDir"], modelname + ".pth"),
                                 config_path=os.path.join(config["ROOT"], hlp_conf["ModelDir"], modelname + ".yml"),
                                 score_thresh=score_thresh)

    start, end = 0, 100
    results = parser.batch_detect(os.path.join(config["ROOT"], config["PDF"]["OutputDir"]), start=start, end=end)
    output_path = os.path.join(config["ROOT"], f"data/dets/{modelname}_{start}_{end}.pkl")
    pickle.dump(results, open(output_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    print("Detection results:")
    print(output_path)

    LayoutBaseParser.verify_saved_dets(output_path,
                                       os.path.join(config["ROOT"], config["PDF"]["OutputDir"]),
                                       "/Users/fan/Documents/Github/ECON2355_data/vis")
