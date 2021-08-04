import numpy as np
import pandas as pd
from collections import Counter


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
    with open("../statistics/num_of_event_cate.txt", 'w', encoding='utf-8') as f:
        f.write("category\ttrain,val,test,total\n")
        f.write("\n".join(statics))


def number_of_event_categories_full(file):
    # 给定的文件中每种事件分别有多少个视频
    categories, cate2idx = read_categories()

    full = pd.read_csv(file, header=0, sep='\t')
    full_labels = full["event_labels"].values
    nums = np.zeros(25, dtype=np.int)

    for label in full_labels:
        labels = label.split(",")
        for l in labels:
            nums[cate2idx[l]] += 1

    statics = ["\t".join([c, str(n)]) for c, n in zip(categories, nums)]
    with open("../statistics/num_of_event_cate_audioset_full.txt", 'w', encoding='utf-8') as f:
        f.write("category\ttotal\n")
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
    with open("../statistics/num_of_event_cate_for_modal.txt", "w", encoding="utf-8") as f:
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

    train_max = 0
    val_max = 0
    test_max = 0

    for labels in train_labels:
        num = len(labels.split(","))
        train_max = max(train_max, num)
        train_nums[num - 1] += 1
    for labels in val_labels:
        num = len(labels.split(","))
        val_max = max(val_max, num)
        val_nums[num - 1] += 1
    for labels in test_labels:
        num = len(labels.split(","))
        test_max = max(test_max, num)
        test_nums[num - 1] += 1

    total_max = max(train_max, val_max, test_max)
    nums = np.arange(total_max) + 1
    train_nums = np.array(train_nums, dtype=np.int)[:total_max]
    val_nums = np.array(val_nums, dtype=np.int)[:total_max]
    test_nums = np.array(test_nums, dtype=np.int)[:total_max]
    print(f"total_max: {total_max}\ntrain_max: {train_max}\nval_max: {val_max}\ntest_max: {test_max}")

    statics = ["\t".join([str(c), ",".join([str(tr), str(va), str(te), str(tr + va + te)])])
               for c, tr, va, te in zip(nums, train_nums, val_nums, test_nums)]
    with open("../statistics/event_cate_num_for_each_video_distribution.txt", 'w', encoding='utf-8') as f:
        f.write("event_cate_number\ttrain,val,test,total\n")
        f.write("\n".join(statics))


def event_cate_num_for_each_video_distribution_full(file):
    # 给定的文件中每个视频发生了多少种事件的统计（横坐标：发生事件种数，纵坐标：视频个数）
    # categories, cate2idx = read_categories()

    total_labels = pd.read_csv(file, header=0, sep='\t')["event_labels"]
    total_nums = [0] * 10
    total_max = 0

    for labels in total_labels:
        num = len(labels.split(","))
        total_max = max(total_max, num)
        total_nums[num - 1] += 1

    nums = np.arange(total_max) + 1
    total_nums = np.array(total_nums, dtype=np.int)[:total_max]
    print(f"total_max: {total_max}\n")

    statics = ["\t".join([str(n), str(t)]) for n, t in zip(nums, total_nums)]
    with open("../statistics/event_cate_num_distribution_audioset.txt", 'w', encoding='utf-8') as f:
        f.write("event_cate_number\ttotal\n")
        f.write("\n".join(statics))


def event_num_modality_for_each_video_distribution():
    # 验证集+测试集每种模态每个视频发生的事件个数分布（横坐标：事件个数（分组audio和visual），纵坐标每个事件个数的视频数）
    # categories, cate2idx = read_categories()
    audio = pd.read_csv("../data/AVVP_eval_audio.csv", header=0, sep='\t')
    visual = pd.read_csv("../data/AVVP_eval_visual.csv", header=0, sep='\t')

    audio_filenames = audio["filename"]
    visual_filenames = visual["filename"]

    audio_dict = Counter(audio_filenames)
    visual_dict = Counter(visual_filenames)

    audio_nums = np.zeros(20, dtype=np.int)
    visual_nums = np.zeros(20, dtype=np.int)

    max_num = 0
    for file in audio_dict:
        num = audio_dict[file]
        audio_nums[num - 1] += 1
        max_num = max(num, max_num)
    for file in visual_dict:
        num = visual_dict[file]
        visual_nums[num - 1] += 1
        max_num = max(num, max_num)

    audio_nums = audio_nums[:max_num]
    visual_nums = visual_nums[:max_num]
    number = np.arange(max_num) + 1

    statics = ["\t".join([str(n), ",".join([str(a), str(v), str(a + v)])])
               for n, a, v in zip(number, audio_nums, visual_nums)]
    with open("../statistics/event_num_modal_distribution.txt", "w", encoding="utf-8") as f:
        f.write("event_number\taudio,visual,total\n")
        f.write("\n".join(statics))


def duration_of_categories_for_modality_distribution():
    # 对于每个模态每种类别的事件发生的平均长度、标准差
    categories, cate2idx = read_categories()

    audio = pd.read_csv("../data/AVVP_eval_audio.csv", header=0, sep='\t')
    visual = pd.read_csv("../data/AVVP_eval_visual.csv", header=0, sep='\t')

    audio_labels = audio["event_labels"]
    visual_labels = visual["event_labels"]

    audio_onset = audio["onset"]
    visual_onset = visual["onset"]
    audio_offset = audio["offset"]
    visual_offset = visual["offset"]

    audio_duration = audio_offset - audio_onset
    visual_duration = visual_offset - visual_onset

    audio_event_duration = [None] * 25
    visual_event_duration = [None] * 25

    for i, d in enumerate(audio_duration):
        label = cate2idx[audio_labels[i]]
        if audio_event_duration[label] is None:
            audio_event_duration[label] = []
        audio_event_duration[label].append(d)
    for i, d in enumerate(visual_duration):
        label = cate2idx[visual_labels[i]]
        if visual_event_duration[label] is None:
            visual_event_duration[label] = []
        visual_event_duration[label].append(d)

    audio_mean = []
    audio_std = []
    visual_mean = []
    visual_std = []
    for cat in range(25):
        audio_mean.append(np.mean(np.array(audio_event_duration[cat])))
        audio_std.append(np.std(np.array(audio_event_duration[cat])))
        visual_mean.append(np.mean(np.array(visual_event_duration[cat])))
        visual_std.append(np.std(np.array(visual_event_duration[cat])))
    audio_mean = np.array(audio_mean)
    visual_mean = np.array(visual_mean)
    audio_std = np.array(audio_std)
    visual_std = np.array(visual_std)

    statics = ["\t".join([str(c), ",".join([str(round(aum, 2)), str(round(vim, 2))]),
                          ",".join([str(round(aus, 2)), str(round(vis, 2))])])
               for c, aum, aus, vim, vis in zip(categories, audio_mean, audio_std, visual_mean, visual_std)]
    with open("../statistics/cate_duration_for_modality_distribution.txt", "w", encoding="utf-8") as f:
        f.write("category\taudio_mean,visual_mean\taudio_std,visual_std\n")
        f.write("\n".join(statics))


if __name__ == '__main__':
    event_cate_num_for_each_video_distribution_full("../data/audioset_remained.csv")
