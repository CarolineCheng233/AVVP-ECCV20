import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_categories():
    categories = []
    with open("../data/categories.txt", "r", encoding='utf-8') as f:
        for line in f:
            cate = line.strip().split("\t")[0]
            categories.append(cate)
    return np.array(categories)


def read_num_of_event_cate():
    train_arr = []
    val_arr = []
    test_arr = []
    total_arr = []
    with open("../statistics/num_of_event_cate.txt", 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            cate, num = line.strip().split("\t")
            train, val, test, total = num.split(",")
            train_arr.append(int(train))
            val_arr.append(int(val))
            test_arr.append(int(test))
            total_arr.append(int(total))
    train_arr = np.array(train_arr).reshape(-1, 1)
    val_arr = np.array(val_arr).reshape(-1, 1)
    test_arr = np.array(test_arr).reshape(-1, 1)
    total_arr = np.array(total_arr).reshape(-1, 1)
    return np.concatenate((train_arr, val_arr, test_arr, total_arr), axis=1), \
           np.array(["train", "val", "test", "total"])


def read_num_of_event_cate_audioset_full():
    total_arr = []
    with open("data/num_of_event_cate_audioset_full.txt", 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            cate, num = line.strip().split("\t")
            total_arr.append(int(num))
    total_arr = np.array(total_arr).reshape(-1, 1)
    return total_arr, np.array(["total"])


def read_num_of_event_cate_for_modal():
    audio_arr = []
    visual_arr = []
    total_arr = []
    with open("data/num_of_event_cate_for_modal.txt", 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            cate, num = line.strip().split("\t")
            audio, visual, total = num.split(",")
            audio_arr.append(int(audio))
            visual_arr.append(int(visual))
            total_arr.append(int(total))
    audio_arr = np.array(audio_arr).reshape(-1, 1)
    visual_arr = np.array(visual_arr).reshape(-1, 1)
    total_arr = np.array(total_arr).reshape(-1, 1)
    return np.concatenate((audio_arr, visual_arr, total_arr), axis=1), \
           np.array(["audio", "visual", "total"])


def read_event_num_modal_distribution():
    # 训练集、验证集、测试集、以及整体视频中每个模态发生事件数量的分布
    audio_arr = []
    visual_arr = []
    total_arr = []
    length = 0
    with open("data/event_num_modal_distribution.txt", 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            length += 1
            cate, num = line.strip().split("\t")
            audio, visual, total = num.split(",")
            audio_arr.append(int(audio))
            visual_arr.append(int(visual))
            total_arr.append(int(total))
    audio_arr = np.array(audio_arr).reshape(-1, 1)
    visual_arr = np.array(visual_arr).reshape(-1, 1)
    total_arr = np.array(total_arr).reshape(-1, 1)
    return np.concatenate((audio_arr, visual_arr, total_arr), axis=1), \
           np.array(["audio", "visual", "total"]), np.arange(length) + 1


def read_event_cate_num_distribution():
    train_arr = []
    val_arr = []
    test_arr = []
    total_arr = []
    length = 0
    with open("data/event_cate_num_for_each_video_distribution.txt", 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            length += 1
            cate, num = line.strip().split("\t")
            train, val, test, total = num.split(",")
            train_arr.append(int(train))
            val_arr.append(int(val))
            test_arr.append(int(test))
            total_arr.append(int(total))
    train_arr = np.array(train_arr).reshape(-1, 1)
    val_arr = np.array(val_arr).reshape(-1, 1)
    test_arr = np.array(test_arr).reshape(-1, 1)
    total_arr = np.array(total_arr).reshape(-1, 1)
    return np.concatenate((train_arr, val_arr, test_arr, total_arr), axis=1), \
           np.array(["train", "val", "test", "total"]), np.arange(length) + 1


def read_event_duration_for_modal():
    # 验证集+测试集、整体视频中每个模态每种类型的事件发生的平均长度、方差
    df = pd.read_csv("data/cate_duration_for_modality_distribution.txt", header=0, sep="\t")
    audio = df["audio_mean,audio_std"].values
    visual = df["visual_mean,visual_std"].values
    total = df["total_mean,total_std"].values

    audio_mean = list()
    audio_std = list()
    visual_mean = list()
    visual_std = list()
    total_mean = list()
    total_std = list()
    for a in audio:
        m, s = a.split(',')
        audio_mean.append(float(m))
        audio_std.append(float(s))
    for v in visual:
        m, s = v.split(',')
        visual_mean.append(float(m))
        visual_std.append(float(s))
    for t in total:
        m, s = t.split(',')
        total_mean.append(float(m))
        total_std.append(float(s))
    return np.array(audio_mean), np.array(audio_std), \
           np.array(visual_mean), np.array(visual_std), \
           np.array(total_mean), np.array(total_std)


def draw_histogram(label, number, names, show_name_per_num=1):
    # number是二维数组，len(number)=len(label)，分组显示，legend和names个数是数组的个数
    assert len(label) == len(number), 'unequal label and number'
    length = len(label)
    fig = plt.figure()
    label = [cat if i % show_name_per_num == 0 else '' for i, cat in enumerate(label)]
    groups = number.shape[1]  # 4
    ew = 0.8 / groups
    x = np.arange(length)
    for g in range(groups):  # 总宽度0.8
        plt.bar(x + ew * g, number[:, g], ew)
    plt.xticks(np.arange(length),
               label,
               fontsize=10,
               fontweight='bold')
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    plt.grid(axis='y', zorder=1)
    plt.legend(names)
    plt.show()


def draw_mean_std(label, number, names, show_name_per_num=1):
    # number三维数组,label是横坐标,number.shape是横坐标数*组数*2(mean和std),names的长度是组数,即每组的名字
    assert len(label) == len(number), 'unequal label and number'
    length = len(label)
    fig = plt.figure()
    label = [cat if i % show_name_per_num == 0 else '' for i, cat in enumerate(label)]
    groups = number.shape[1]  # 3,audio,visual,total
    ew = 0.8 / groups
    x = np.arange(length)
    for g in range(groups):  # 总宽度0.8
        plt.bar(x + ew * g, number[:, g, 0], width=ew, yerr=number[:, g, 1])  # 添加errorbar
    plt.xticks(np.arange(length),
               label,
               fontsize=10,
               fontweight='bold')
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    plt.grid(axis='y', zorder=1)
    plt.legend(names)
    plt.show()


if __name__ == '__main__':
    array, names = read_num_of_event_cate()
    label = read_categories()
    draw_histogram(label, array, names)
