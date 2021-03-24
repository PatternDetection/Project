import sys

sys.path.append("../cv")

import os
import pickle

import yaml
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt

from preprocess import batch_convert_pdfs_to_jpgs
from layout_parser import HarvardLayoutParser
from OCR import BaiduOCR

if __name__ == '__main__':
    with open("../config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    taskId = "cqy"
    use_cached_dets = True

    # S0: PDF --> JPEG
    # batch_convert_pdfs_to_jpgs(os.path.join(config["ROOT"], config["PDF"]["InputDir"]),
    #                            os.path.join(config["ROOT"], config["PDF"]["OutputDir"]))

    # S1: detect bbxes
    hlp_conf = config["LAYOUT"]["HarvardLP"]
    dataset = "Prima"
    modelname = hlp_conf["OpenModels"][dataset][0]
    score_thresh = 0.9
    parser = HarvardLayoutParser(dataset,
                                 model_path=os.path.join(config["ROOT"], hlp_conf["ModelDir"], modelname + ".pth"),
                                 config_path=os.path.join(config["ROOT"], hlp_conf["ModelDir"], modelname + ".yml"),
                                 score_thresh=score_thresh)

    output_path = os.path.join(config["ROOT"], f"data/dets/{taskId}_{modelname}.pkl")

    if not use_cached_dets or not os.path.exists(output_path):
        results = parser.batch_detect(os.path.join(config["ROOT"], config["PDF"]["OutputDir"]), start=-1, end=-1)
        pickle.dump(results, open(output_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    else:
        results = pickle.load(open(output_path, "rb"))

    # parser.verify_saved_dets(output_path,
    #                          os.path.join(config["ROOT"], config["PDF"]["OutputDir"]),
    #                          os.path.join(config["ROOT"], "data/vis"))

    API_KEY = config["OCR"]["APIKey"]
    SECRET_KEY = config["OCR"]["SecretKey"]
    bocr = BaiduOCR(API_KEY, SECRET_KEY)

    txt_output_dir = os.path.join(config["ROOT"], config["OCR"]["OutputDir"])

    n_bbxes = 0
    n_valid_bbxes = 0

    for relpath, text_bbxes in tqdm(results):
        abspath = os.path.join(config["ROOT"], config["PDF"]["OutputDir"], relpath)

        im = cv2.imread(abspath)

        n = len(text_bbxes)
        text_bbxes.sort(key=lambda x: x[1])

        with open(os.path.join(txt_output_dir, os.path.basename(relpath).replace("jpg", "txt")), "w") as f:
            n_bbxes += len(text_bbxes)

            for i, x in enumerate(text_bbxes):
                l, t, r, b, _, score = x

                # Filter small bbx out
                if (r - l) < 0.5:
                    continue

                n_valid_bbxes += 1

                c = parser.crop(im, [max(l - 0.02, 0), t, min(r + 0.02, 1), b])
                try:
                    resp = bocr.query_cv2im(c)
                    f.write("".join(resp) + "\n")
                except:
                    print("error")
                    plt.imshow(c)
                    plt.show()

    text_bbxes = parser.detect(im, keep_text_only=True, dump_to_tuples=True)

    print(f"{n_bbxes} bbxes detected, {n_valid_bbxes} kept.")
