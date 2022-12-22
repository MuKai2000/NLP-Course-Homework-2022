import os
import random
import xml.etree.ElementTree as ET
import jieba

def clean(en, zh):

    # print(en, zh)
    # 去除相同语句对
    if en == zh:
        return 1,en, zh
    # 去除网站
    if en[:4] == "http":
        return 1, en, zh
    
    # 分词
    en_split = en.split(' ')
    zh_split = jieba.lcut(zh, cut_all=False)

    # 去除长度差距过大的
    rate = 3
    if float(len(en_split))/float(len(zh_split)) > rate  or float(len(en_split))/float(len(zh_split)) < (1.00 / rate):
        return 1, en, zh


    return 0, en, zh


def main():
    file_path = "./en-zh.tmx"
    tree = ET.parse(file_path)
    root = tree.getroot()
    body = root[1]

    file_en = open('./data/raw.en', 'w')
    file_zh = open('./data/raw.zh', 'w')

    with open('./target.txt', 'w') as txtfile:
        for item in body:
            en = item[0][0].text
            zh = item[1][0].text
            tag, en, zh = clean(en, zh)
            if tag == 1:
                continue
            txtfile.write(en + "\t" + zh + "\n")
            file_en.write(en + "\n")
            file_zh.write(zh + "\n")

    file_en.close()
    file_zh.close()


if __name__ == "__main__":
    main()