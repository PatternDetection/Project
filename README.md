# Data Extraction and Sentimental Analysis from Equity Research Reports at Scale

## Econ 2355, Unleashing Novel Data at Scale, Spring 2021

### Group Members

* Chenfan Zhuang
* Xin Zeng
* Qinyi Chen
* Yujie Cai

## Installation
Please read 'requirements.txt' for details. 

## Usage
Demo of CV module can be found in `./notebooks/`. To process files in batch, please follow the steps:

* Firstly preprocess data, run `code/cv/convert_pdf_to_jpg.py`
* To test the public models, please:
    * Run `code/cv/download_models.jpy`
    * Run `code/cv/parse_layout.jpy`
    * Call functions in `code/cv/evaluate.py`. `notebooks/evaluate.ipynb` is recommended to see the usage.
* To finetune model, please see codes in `code/cv/layout5_detectron2.ipynb`.
* Finally, run `code/cv/OCR.py` to extract characters, but `API_KEY` and `API_SECRET` is required.
