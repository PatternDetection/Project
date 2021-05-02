# -*- coding: utf-8 -*-

import pandas as pd
import os
import numpy as np

stock_dir = '/Users/zengxin/Study/Econ2355/data'
senti_dir = '/Users/zengxin/Study/Econ2355/result'


def calc_return(ticker, date, freq=5):
    # preprocess for different exchange markets in China
    if ticker[:1] == "6":
        file_dir = os.path.join(stock_dir, "%s.SH.csv" % ticker)
    else:
        file_dir = os.path.join(stock_dir, "%s.SZ.csv" % ticker)
    if not os.path.exists(file_dir):
        return ticker
    data = pd.read_csv(file_dir)
    index_curr = np.argmax(data['日期'].values >= float(date))
    if index_curr + freq >= data.shape[0]:
        return ticker
    ret = (data['收盘价(元)'].values[index_curr+freq] - data['收盘价(元)'].values[index_curr]) / \
          data['收盘价(元)'].values[index_curr]
    return ret


def calc_corr(freq_list=[1, 5, 10, 20, 30, 60, 100, 200]):
    #senti_df = pd.read_csv(os.path.join(senti_dir, "sentiment_finbert_full_final.csv"), index_col="Unnamed: 0")
    senti_df = pd.read_csv(os.path.join(senti_dir, "sentiment_finbert_full_final.csv"))
    senti_df = senti_df.groupby(["file_name"]).mean()
    senti_df = senti_df.reset_index()
    senti_df[["ticker", "date", "redun"]] = senti_df["file_name"].str.split('-', expand=True)
    senti_df = senti_df.drop(["redun", "file_name"], axis=1)
    for freq in freq_list:
        senti_df["return_" + str(freq)] = np.nan
        for i in range(senti_df.shape[0]):
            ret = calc_return(senti_df["ticker"].values[i], senti_df["date"].values[i], freq)
            if type(ret) is not str:
                senti_df.loc[i, "return_" + str(freq)] = ret
    return senti_df


if __name__ == "__main__":
    calc_corr().to_csv('/Users/zengxin/Study/Econ2355/result/corr_3.csv')