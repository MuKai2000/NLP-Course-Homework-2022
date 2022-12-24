import os
import random
import xml.etree.ElementTree as ET
import jieba
import time

def clean(en, zh, rand_rate=0.7):

    # print(en, zh)
    # 去除相同语句对
    if en == zh:
        return 1, en, zh
    # 去除网站
    if en[:4] == "http":
        return 1, en, zh
    
    # 分词
    en_split = en.split(' ')
    zh_split = jieba.lcut(zh, cut_all=False)

    # 去除过短或过长
    if len(en_split) < 5 or len(en_split) > 200:
        return 1, en, zh
    if len(zh_split) < 5 or len(zh_split) > 200:
        return 1, en, zh

    # 去除长度差距过大的
    rate = 2.5
    if float(len(en_split))/float(len(zh_split)) > rate  or float(len(en_split))/float(len(zh_split)) < (1.00 / rate):
        # print(len(en_split),len(zh_split))
        return 1, en, zh

    # 去除特殊单词
    del_list = ['大笑','笑声','鼓掌', '(', '（', '[', '�', '–', '—']
    for word in del_list:
        if word in zh_split:
            return 1, en, zh
        if word in en_split:
            return 1, en, zh

    # 随机去除
    # print(random.random())
    seed = random.random()
    if seed >= rand_rate:
        return 1, en, zh
        pass

    return 0, en, zh


def main():

    file_en = open('./data/raw.en', 'w')
    file_zh = open('./data/raw.zh', 'w')
    txtfile = open('./target.txt', 'w')

    """file_org_en = open('./en.txt', 'r').readlines()
    file_org_zh = open('./zh.txt', 'r').readlines().strip('\n')"""

    data = open('./news-commentary-v15.en-zh.tsv','r').readlines()

    num_seq = 0
    num_org = 0
    
    # print(len(data))
    for i in range(len(data)):
        d = data[i].strip('\n').split('\t')
        en, zh = d[0].strip(' '), d[1].strip(' ')
        # print(en, zh)
        num_org += 1
        tag, en, zh = clean(en, zh, 0.38)
        if tag != 1:
            txtfile.write(en + "\t" + zh + "\n")
            file_en.write(en + "\n")
            file_zh.write(zh + "\n")
            num_seq += 1
    
    print(num_org)
    print(num_seq)

    file_en.close()
    file_zh.close()


if __name__ == "__main__":
    main()