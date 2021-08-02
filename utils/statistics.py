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
        f.write("category\taudio,visual,total\n")
        f.write("\n".join(statics))


def event_cate_num_for_each_video_distribution():
    # 训练集、验证集、测试集、以及整体视频中每个视频发生了多少种事件的统计（横坐标：发生事件种数，纵坐标：视频个数）
    # categories, cate2idx = read_categories()

    train_labels = pd.read_csv("../data/AVVP_train.csv", header=0, sep='\t')["event_labels"]
    val_labels = pd.read_csv("../data/AVVP_val_pd.csv", header=0, sep='\t')["event_labels"]
    test_labels = pd.read_csv("../data/AVVP_test_pd.csv", header=0, sep='\t')["event_labels"]

    train_nums = [0] * 10
    val_nums = [0] * 10
    test_nums = [0] * 10
    # total_nums = [0] * 10

    train_max = 0
    val_max = 0
    test_max = 0
    # total_max = 0

    for labels in train_labels:
        num = len(labels.split(","))
        train_max = max(train_max, num)
        # total_max = max(total_max, num)
        train_nums[num - 1] += 1
        # total_nums[num - 1] += 1
    for labels in val_labels:
        num = len(labels.split(","))
        val_max = max(val_max, num)
        # total_max = max(total_max, num)
        val_nums[num - 1] += 1
        # total_nums[num - 1] += 1
    for labels in test_labels:
        num = len(labels.split(","))
        test_max = max(test_max, num)
        # total_max = max(total_max, num)
        test_nums[num - 1] += 1
        # total_nums[num - 1] += 1

    total_max = max(train_max, val_max, test_max)
    nums = np.arange(total_max) + 1
    train_nums = np.array(train_nums, dtype=np.int)[:total_max]
    val_nums = np.array(val_nums, dtype=np.int)[:total_max]
    test_nums = np.array(test_nums, dtype=np.int)[:total_max]
    print(f"total_max: {total_max}\ntrain_max: {train_max}\nval_max: {val_max}\ntest_max: {test_max}")

    statics = ["\t".join([str(c), ",".join([str(tr), str(va), str(te), str(tr + va + te)])])
               for c, tr, va, te in zip(nums, train_nums, val_nums, test_nums)]
    with open("../data/event_cate_num_for_each_video_distribution.txt", 'w', encoding='utf-8') as f:
        f.write("event_cate_number\ttrain,val,test,total\n")
        f.write("\n".join(statics))


if __name__ == '__main__':
    event_cate_num_for_each_video_distribution()
