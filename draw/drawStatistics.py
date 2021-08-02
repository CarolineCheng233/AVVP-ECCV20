import numpy as np
import matplotlib.pyplot as plt


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
    with open("../data/num_of_event_cate.txt", 'r', encoding='utf-8') as f:
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


if __name__ == '__main__':
    array, names = read_num_of_event_cate()
    label = read_categories()
    draw_histogram(label, array, names)
