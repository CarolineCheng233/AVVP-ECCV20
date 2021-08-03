import numpy as np
import pandas as pd


def read_cates():
    file = "../audioset/idx_mid_class.csv"
    df = pd.read_csv(file, header=0, sep=',')
    mid = df["mid"]
    class_name = df["display_name"]
    mid2class = dict()
    for m, c in zip(mid, class_name):
        mid2class[m] = c
    return mid, mid2class


def filter_videos(file, mids, mid2class, wfile):
    mids = set(mids)
    df = pd.read_csv(file, header=2, sep=", ", engine='python')
    videos = df["# YTID"].values
    start_seconds = df["start_seconds"].values
    end_seconds = df["end_seconds"].values
    positive_labels = df["positive_labels"].values

    start_seconds = start_seconds.astype(np.float).astype(np.int)
    end_seconds = end_seconds.astype(np.float).astype(np.int)

    remained_videos = list()
    for i, v in enumerate(videos):
        labels = set(positive_labels[i][1:-1].split(","))
        if len(mids & labels) > 0:
            mids = list(mids & labels)
            classes = list()
            for m in mids:
                classes.append(mid2class[m])
            name = v + "_" + str(start_seconds[i]) + "_" + str(end_seconds[i])
            remained_videos.append("\t".join([name, ",".join(classes)]))

    with open(wfile, 'w', encoding='utf-8') as f:
        f.write("filename\tevent_labels\n")
        f.write("\n".join(remained_videos))


if __name__ == '__main__':
    mid, mid2class = read_cates()
    filter_videos("../audioset/eval_segments.csv", mid, mid2class, "../audioset/AVVP_addi_eval.csv")
