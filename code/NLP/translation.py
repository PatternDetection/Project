# -*- coding: utf-8 -*-

from googletrans import Translator
import os


def translate_batch(source_dir, target_dir):
    for dirpath, dirnames, filenames in os.walk(source_dir):
        for subfile in filenames:
            file_name = os.path.join(source_dir, subfile)
            if file_name[-3:] != "txt":
                continue
            out_file = os.path.join(target_dir, subfile)
            if os.path.exists(out_file):
                continue
            print(file_name)
            with open(file_name, encoding="utf8") as f:
                inp = f.readlines()
            if inp:
                inp = "".join(inp)
                translator = Translator()
                oup = translator.translate(inp, src='zh-cn').text
            else:
                oup = ""
            text_file = open(out_file, "w")
            n = text_file.write(oup)
            text_file.close()


if __name__ == "__main__":
    # test for translator API
    translator = Translator()
    print(translator.translate('全年销售额同増89%至5028.5亿,超额完成销售目标,克尔瑞排名第五。公司2020年全年实现签约金额5028.5亿元,同比增长8.9%;签约面积3409.2万平方米,同比增长9.2%,销售规模位列行业第五;销售均价47497元/平方米,同比降低03%'
                     '拿地金额同增61%至1765亿,一二线布局力度加大,拿地均价同增28%至5540元。公司2020年全年拿地金额1765.0亿元,同比增长61.0%,为同期销售金额的35.1%;拿地面积3185.7万平方米,同比上升25.8%,为同期销售面积的93.4%;新増权益建筑面积2209.5万方,权益比69.4%拿地均价5540.4元/平方米,同比增长28.0%,地售比37.6%,较上年同其增长8.3个百分点。公司2020年全年一、二、三线城市拿地面积同比分另增长11.7%、45.1%、7.0%,占总比分别为58%、46.5%和477%,二线城市拿地规模増长显著;分堿市群来看主要集中在粤港澳、海峡西岸和长江'
                     '前三季度归母净利同增29%,预收账款及合同负债覆盖倍数1.61,锁定业绩释放。2020年前三季度,公司实现营业收入1174.04亿元,同比增长5.0%;实现归母净利润为132.0亿元,同比增长2.9%。毛利率同比下降1.9个百分点至34.0%,下降情况好于预期。销售回笼3317.4亿元,回笼率达90.3%。三季度预收款项加合同负债3801.1亿元,较2019年末增长'
                     '融资成本优势明显(2020H1为484%),债务指标稳守三条红线。截至2020年9月30日,公司总资产规模11479.0亿元,同比增长17.7%;净资产规模2531.6亿元,同比增长27.5%。公司上半年综合融资成本4.84%,同比下降0.11pct,三季度末资产负债率为78.0%,较上年同期减少1.7个百分点;净负债率为70.87%,较上年同期减少11.4个百分点,公司净负债率显著改善;剔除预收账款的资产负债率为67.0%,较去年同期増长0.54'
                     '投资建议:我们预测公司2020/2021/2022年收入增速为154%/20.2%/14.3%,归母净利润为312.8/376.1/411.6亿元,增速分别为11.9%/20.3%/9.4%。2020年动态PE为60X,维持“买入”评级'
                     '风险提示:结算进度不及预期。毛利率下降超预期。行业及融资政策收紧预期。疫情影响超预期', src='zh-cn'))
    # conduct translation in batch for all our files
    source_dir = '/Users/zengxin/Study/Econ2355/OCR'
    target_dir = '/Users/zengxin/Study/Econ2355/OCR_English'
    translate_batch(source_dir, target_dir)