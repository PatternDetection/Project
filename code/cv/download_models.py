# ModelZoo provided in
# https://github.com/Layout-Parser/layout-parser/blob/master/src/layoutparser/models/catalog.py
import requests
import os

MODEL_CATALOG = {
    'HJDataset': {
        'faster_rcnn_R_50_FPN_3x': 'https://www.dropbox.com/s/6icw6at8m28a2ho/model_final.pth?dl=1',
        'mask_rcnn_R_50_FPN_3x': 'https://www.dropbox.com/s/893paxpy5suvlx9/model_final.pth?dl=1',
        'retinanet_R_50_FPN_3x': 'https://www.dropbox.com/s/yxsloxu3djt456i/model_final.pth?dl=1'
    },
    "PubLayNet": {
        "faster_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/dgy9c10wykk4lq4/model_final.pth?dl=1",
        "mask_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/d9fc9tahfzyl6df/model_final.pth?dl=1",
        "mask_rcnn_X_101_32x8d_FPN_3x": "https://www.dropbox.com/s/57zjbwv6gh3srry/model_final.pth?dl=1"
    },
    "PrimaLayout": {
        "mask_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/h7th27jfv19rxiy/model_final.pth?dl=1"
    },
    "NewspaperNavigator": {
        'faster_rcnn_R_50_FPN_3x': 'https://www.dropbox.com/s/6ewh6g8rqt2ev3a/model_final.pth?dl=1',
    }
}

CONFIG_CATALOG = {
    'HJDataset': {
        'faster_rcnn_R_50_FPN_3x': 'https://www.dropbox.com/s/j4yseny2u0hn22r/config.yml?dl=1',
        'mask_rcnn_R_50_FPN_3x': 'https://www.dropbox.com/s/4jmr3xanmxmjcf8/config.yml?dl=1',
        'retinanet_R_50_FPN_3x': 'https://www.dropbox.com/s/z8a8ywozuyc5c2x/config.yml?dl=1'
    },
    "PubLayNet": {
        "faster_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/f3b12qc4hc0yh4m/config.yml?dl=1",
        "mask_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/u9wbsfwz4y0ziki/config.yml?dl=1",
        "mask_rcnn_X_101_32x8d_FPN_3x": "https://www.dropbox.com/s/nau5ut6zgthunil/config.yaml?dl=1"
    },
    "PrimaLayout": {
        "mask_rcnn_R_50_FPN_3x": "https://www.dropbox.com/s/yc92x97k50abynt/config.yaml?dl=1"
    },
    "NewspaperNavigator": {
        'faster_rcnn_R_50_FPN_3x': 'https://www.dropbox.com/s/wnido8pk4oubyzr/config.yml?dl=1',
    }
}


def download_file(url, savepath):
    r = requests.get(url, stream=True)
    with open(savepath, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()

    print(f"Download {url} to {savepath}")


if __name__ == '__main__':
    savedir = "../../models/"

    for dataset in CONFIG_CATALOG:
        options = CONFIG_CATALOG[dataset]

        subdir = os.path.join(savedir, dataset)
        os.makedirs(subdir, exist_ok=True)

        for name in options:
            config_url = options[name]
            model_url = MODEL_CATALOG[dataset][name]

            download_file(config_url, os.path.join(subdir, f"{name}.config"))
            download_file(model_url, os.path.join(subdir, f"{name}.pth"))
