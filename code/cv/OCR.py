"""
Code reference: https://cloud.baidu.com/doc/OCR/s/dk3iqnq51?_=1615343929412
"""
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import os
import json
import base64
from urllib.request import urlopen
from urllib.request import Request
from urllib.parse import urlencode

import yaml
import cv2


class BaiduOCR(object):
    # OCR_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic"
    OCR_URL = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"


    TOKEN_URL = 'https://aip.baidubce.com/oauth/2.0/token'

    def __init__(self, API_KEY=None, SECRET_KEY=None, token=None):
        self.token = token or self.fetch_token(API_KEY, SECRET_KEY)

    @staticmethod
    def fetch_token(API_KEY, SECRET_KEY):
        params = {'grant_type': 'client_credentials',
                  'client_id': API_KEY,
                  'client_secret': SECRET_KEY}

        post_data = urlencode(params).encode('utf-8')
        req = Request(BaiduOCR.TOKEN_URL, post_data)
        f = urlopen(req, timeout=5)
        result_str = f.read().decode()
        result = json.loads(result_str)
        return result['access_token']

    @staticmethod
    def request(token, im):
        url = BaiduOCR.OCR_URL + "?access_token=" + token
        data = urlencode({'image': base64.b64encode(im)}).encode('utf-8')
        req = Request(url, data)
        f = urlopen(req)
        resp = json.loads(f.read().decode())

        try:
            return [r["words"] for r in resp["words_result"]]
        except:
            print(resp)

    def query_filepath(self, path):
        im = open(path, "rb").read()
        return self.request(self.token, im)

    def query_cv2im(self, im):
        _, im_arr = cv2.imencode('.jpg', im)  # im_arr: image in Numpy one-dim array format.
        im_bytes = im_arr.tobytes()
        return self.request(self.token, im_bytes)


if __name__ == '__main__':
    with open("../config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    API_KEY = config["OCR"]["APIKey"]
    SECRET_KEY = config["OCR"]["SecretKey"]

    b = BaiduOCR(API_KEY, SECRET_KEY)
    # Or specify a token
    # b = BaiduOCR(None, None, token)

    path = "../../notebooks/ocr.png"
    resp = b.query_filepath(path)

    # Or pass in a cv2 image
    # im = cv2.imread(path)
    # resp = b.query_cv2im(im)
    print(resp)
