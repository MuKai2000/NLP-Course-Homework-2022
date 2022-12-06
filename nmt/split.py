import sys
import random


def split(src, src_path, tgt, tgt_path, save_dir="", ratio=(0.9, 0.05, 0.05)):
    
    print(src, src_path, tgt, tgt_path, save_dir, ratio)
    
    # read file
    d_src = open(src_path, encoding='utf-8').readlines()
    d_tgt = open(tgt_path, encoding='utf-8').readlines()
    
    src_train = open(save_dir + 'train.' + src, 'w', encoding='utf-8')
    src_test = open(save_dir + 'test.' + src, 'w', encoding='utf-8')
    src_val = open(save_dir + 'valid.' + src, 'w', encoding='utf-8')
  
    tgt_train = open(save_dir + 'train.' + tgt, 'w', encoding='utf-8')
    tgt_test = open(save_dir + 'test.' + tgt, 'w', encoding='utf-8')
    tgt_val = open(save_dir + 'valid.' + tgt, 'w', encoding='utf-8')

    for s, t in zip(d_src, d_tgt):
        rand = random.random()
        # print(type(s), type(t),rand,"--------")
        if 0 < rand <= ratio[0]:
            src_train.write(s)
            tgt_train.write(t)
        elif ratio[0] < rand <= ratio[0] + ratio[1]:
            src_test.write(s)
            tgt_test.write(t)
        else:
            src_val.write(s)
            tgt_val.write(t)

    src_train.close()
    src_test.close()
    src_val.close()
    tgt_train.close()
    tgt_test.close()
    tgt_val.close()








if __name__=='__main__':
    # test
    # src_data = "/home/koukq0907/nmt/en-zh/niutrans-smt-sample/data/clean.en"
    # tgt_data = "/home/koukq0907/nmt/en-zh/niutrans-smt-sample/data/clean.zh"
    # save_dir = "/home/koukq0907/nmt/en-zh/niutrans-smt-sample/data/"
    # split(src_data, tgt_data, "en", "zh")

    split(src=sys.argv[1], src_path=sys.argv[2], tgt=sys.argv[3], tgt_path=sys.argv[4], save_dir=sys.argv[5], ratio=(0.95, 0.025, 0.025))