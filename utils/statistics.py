import numpy as np
import pandas as pd


# data/categories.txt


def read_categories():
    categories = []
    cate2idx = dict()
    with open("../data/categories.txt", "r", encoding='utf-8') as f:
        for line in f:
            cate = line.strip().split("\t")[0]
            categories.append(cate)
            cate2idx[cate] = len(cate2idx)
    categories = np.array(categories)
    return categories, cate2idx


def number_of_event_categories():
    # 训练集、验证集、测试集、以及整体每种事件分别有多少个视频
    categories, cate2idx = read_categories()

    train = pd.read_csv("../data/AVVP_train.csv", header=0, sep='\t')
    val = pd.read_csv("../data/AVVP_val_pd.csv", header=0, sep='\t')
    test = pd.read_csv("../data/AVVP_test_pd.csv", header=0, sep='\t')

    train_labels = train["event_labels"]
    val_labels = val["event_labels"]
    test_labels = test["event_labels"]

    train_nums = np.zeros(25, dtype=np.int)
    val_nums = np.zeros(25, dtype=np.int)
    test_nums = np.zeros(25, dtype=np.int)

    for i in range(len(train_labels)):
        labels = train_labels[i].split(",")
        for l in labels:
            train_nums[cate2idx[l]] += 1

    for i in range(len(val_labels)):
        labels = val_labels[i].split(",")
        for l in labels:
            val_nums[cate2idx[l]] += 1

    for i in range(len(test_labels)):
        labels = test_labels[i].split(",")
        for l in labels:
            test_nums[cate2idx[l]] += 1

    statics = ["\t".join([c, ",".join([str(tr), str(va), str(te), str(tr + va + te)])])
               for c, tr, va, te in zip(categories, train_nums, val_nums, test_nums)]
    with open("../data/num_of_event_cate.txt", 'w', encoding='utf-8') as f:
        f.write("category\ttrain,val,test,total\n")
        f.write("\n".join(statics))


def number_of_event_categories_for_modality():
    # 验证集+测试集每个模态对于每种事件发生的次数
    categories, cate2idx = read_categories()
    audio_labels = pd.read_csv("../data/AVVP_eval_audio.csv", header=0, sep='\t')["event_labels"]
    visual_labels = pd.read_csv("../data/AVVP_eval_visual.csv", header=0, sep='\t')["event_labels"]

    audio_nums = np.zeros(25, dtype=np.int)
    visual_nums = np.zeros(25, dtype=np.int)

    for i in range(len(audio_labels)):
        label = audio_labels[i]
        audio_nums[cate2idx[label]] += 1

    for i in range(len(visual_labels)):
        label = visual_labels[i]
        visual_nums[cate2idx[label]] += 1

    statics = ["\t".join([c, ",".join([str(au), str(vi), str(au + vi)])])
               for c, au, vi in zip(categories, audio_nums, visual_nums)]
    with open("../data/num_of_event_cate_for_modal.txt", "w", encoding="utf-8") as f:
        f.write("category\taudio,visual\n")
        f.write("\n".join(statics))


if __name__ == '__main__':
    number_of_event_categories_for_modality()
