import numpy as np
# from sklearn.metrics import confusion_matrix


def Precision(X_pre, X_gt):
    N = len(X_pre)
    p = 0.0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += np.sum(x * y) / np.sum(x)
    return p / N


def Recall(X_pre, X_gt):
    N = len(X_pre)
    p = 0.0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += np.sum(x * y) / np.sum(y)
    return p / N


def F1(X_pre, X_gt):
    N = len(X_pre)
    p = 0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += 2 * np.sum(x * y) / (np.sum(x) + np.sum(y))
    return p / N


def event_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av):
    # extract events
    N = 25
    event_p_a = [None for n in range(25)]
    event_gt_a = [None for n in range(25)]
    event_p_v = [None for n in range(25)]
    event_gt_v = [None for n in range(25)]
    event_p_av = [None for n in range(25)]
    event_gt_av = [None for n in range(25)]

    TP_a = np.zeros(25)
    TP_v = np.zeros(25)
    TP_av = np.zeros(25)

    FP_a = np.zeros(25)
    FP_v = np.zeros(25)
    FP_av = np.zeros(25)

    FN_a = np.zeros(25)
    FN_v = np.zeros(25)
    FN_av = np.zeros(25)

    for n in range(N):  # 对每个类别分别计算
        seq_pred = SO_a[n, :]
        if np.sum(seq_pred) != 0:
            x = extract_event(seq_pred, n)  # 每个event分别提取出来一个array
            event_p_a[n] = x
        seq_gt = GT_a[n, :]
        if np.sum(seq_gt) != 0:
            x = extract_event(seq_gt, n)
            event_gt_a[n] = x

        seq_pred = SO_v[n, :]
        if np.sum(seq_pred) != 0:
            x = extract_event(seq_pred, n)
            event_p_v[n] = x
        seq_gt = GT_v[n, :]
        if np.sum(seq_gt) != 0:
            x = extract_event(seq_gt, n)
            event_gt_v[n] = x

        seq_pred = SO_av[n, :]
        if np.sum(seq_pred) != 0:
            x = extract_event(seq_pred, n)
            event_p_av[n] = x
        seq_gt = GT_av[n, :]
        if np.sum(seq_gt) != 0:
            x = extract_event(seq_gt, n)
            event_gt_av[n] = x

        tp, fp, fn = event_wise_metric(event_p_a[n], event_gt_a[n])
        TP_a[n] += tp
        FP_a[n] += fp
        FN_a[n] += fn

        tp, fp, fn = event_wise_metric(event_p_v[n], event_gt_v[n])
        TP_v[n] += tp
        FP_v[n] += fp
        FN_v[n] += fn

        tp, fp, fn = event_wise_metric(event_p_av[n], event_gt_av[n])
        TP_av[n] += tp
        FP_av[n] += fp
        FN_av[n] += fn

    TP = TP_a + TP_v
    FN = FN_a + FN_v
    FP = FP_a + FP_v

    n = len(FP_a)
    F_a = []
    for ii in range(n):
        if (TP_a + FP_a)[ii] != 0 or (TP_a + FN_a)[ii] != 0:
            F_a.append(2 * TP_a[ii] / (2 * TP_a[ii] + (FN_a + FP_a)[ii]))

    F_v = []
    for ii in range(n):
        if (TP_v + FP_v)[ii] != 0 or (TP_v + FN_v)[ii] != 0:
            F_v.append(2 * TP_v[ii] / (2 * TP_v[ii] + (FN_v + FP_v)[ii]))

    F = []
    for ii in range(n):
        if (TP + FP)[ii] != 0 or (TP + FN)[ii] != 0:
            F.append(2 * TP[ii] / (2 * TP[ii] + (FN + FP)[ii]))

    F_av = []
    for ii in range(n):
        if (TP_av + FP_av)[ii] != 0 or (TP_av + FN_av)[ii] != 0:
            F_av.append(2 * TP_av[ii] / (2 * TP_av[ii] + (FN_av + FP_av)[ii]))

    if len(F_a) == 0:  # 该视频没有audio事件发生且没有预测任何audio事件
        f_a = 1.0  # all true negatives
    else:
        f_a = (sum(F_a) / len(F_a))

    if len(F_v) == 0:  # 该视频没有visual事件发生且没有预测任何visual事件
        f_v = 1.0  # all true negatives
    else:
        f_v = (sum(F_v) / len(F_v))

    if len(F) == 0:  # 该视频没有任何事件发生—不会出现这种情况
        f = 1.0  # all true negatives
    else:
        f = (sum(F) / len(F))
    if len(F_av) == 0:  # 该视频没有任何audio-visual事件发生且没有预测任何audio-visual事件
        f_av = 1.0  # all true negatives
    else:
        f_av = (sum(F_av) / len(F_av))

    return f_a, f_v, f, f_av


def segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av):
    # compute F scores
    # audio
    TP_a = np.sum(SO_a * GT_a, axis=1)  # 25
    FN_a = np.sum((1 - SO_a) * GT_a, axis=1)  # 25
    FP_a = np.sum(SO_a * (1 - GT_a), axis=1)  # 25

    n = len(FP_a)
    F_a = []
    for ii in range(n):  # 对于每个类别计算F-score
        if (TP_a + FP_a)[ii] != 0 or (TP_a + FN_a)[ii] != 0:
            F_a.append(2 * TP_a[ii] / (2 * TP_a[ii] + (FN_a + FP_a)[ii]))

    # visual
    TP_v = np.sum(SO_v * GT_v, axis=1)
    FN_v = np.sum((1 - SO_v) * GT_v, axis=1)
    FP_v = np.sum(SO_v * (1 - GT_v), axis=1)
    F_v = []
    for ii in range(n):
        if (TP_v + FP_v)[ii] != 0 or (TP_v + FN_v)[ii] != 0:
            F_v.append(2 * TP_v[ii] / (2 * TP_v[ii] + (FN_v + FP_v)[ii]))

    # audio-visual, 并集
    TP = TP_a + TP_v
    FN = FN_a + FN_v
    FP = FP_a + FP_v

    n = len(FP)

    F = []
    for ii in range(n):
        if (TP + FP)[ii] != 0 or (TP + FN)[ii] != 0:
            F.append(2 * TP[ii] / (2 * TP[ii] + (FN + FP)[ii]))

    # audio-visual, 交集
    TP_av = np.sum(SO_av * GT_av, axis=1)
    FN_av = np.sum((1 - SO_av) * GT_av, axis=1)
    FP_av = np.sum(SO_av * (1 - GT_av), axis=1)
    n = len(FP_av)
    F_av = []
    for ii in range(n):
        if (TP_av + FP_av)[ii] != 0 or (TP_av + FN_av)[ii] != 0:
            F_av.append(2 * TP_av[ii] / (2 * TP_av[ii] + (FN_av + FP_av)[ii]))

    if len(F_a) == 0:  # 该视频没有audio事件发生并且也没有预测出任何audio事件
        f_a = 1.0  # all true negatives
    else:
        f_a = (sum(F_a) / len(F_a))

    if len(F_v) == 0:  # 该视频没有visual事件发生并且也没有预测出任何visual事件
        f_v = 1.0  # all true negatives
    else:
        f_v = (sum(F_v) / len(F_v))

    if len(F) == 0:  # 没有任何audio或visual事件发生-不会有这种情况
        f = 1.0  # all true negatives
    else:
        f = (sum(F) / len(F))
    if len(F_av) == 0:  # 没有任何audio-visual事件(交集)并且也没有预测出任何audio-visual交集事件
        f_av = 1.0  # all true negatives
    else:
        f_av = (sum(F_av) / len(F_av))

    return f_a, f_v, f, f_av


def event_level_by_cat(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av):
    # 统计每个类别的F-score
    # extract events
    N = 25
    event_p_a = [None for n in range(25)]
    event_gt_a = [None for n in range(25)]
    event_p_v = [None for n in range(25)]
    event_gt_v = [None for n in range(25)]
    event_p_av = [None for n in range(25)]
    event_gt_av = [None for n in range(25)]

    TP_a = np.zeros(25)
    TP_v = np.zeros(25)
    TP_av = np.zeros(25)

    FP_a = np.zeros(25)
    FP_v = np.zeros(25)
    FP_av = np.zeros(25)

    FN_a = np.zeros(25)
    FN_v = np.zeros(25)
    FN_av = np.zeros(25)

    for n in range(N):  # 对每个类别分别计算
        seq_pred = SO_a[n, :]
        if np.sum(seq_pred) != 0:
            x = extract_event(seq_pred, n)  # 每个event分别提取出来一个array
            event_p_a[n] = x
        seq_gt = GT_a[n, :]
        if np.sum(seq_gt) != 0:
            x = extract_event(seq_gt, n)
            event_gt_a[n] = x

        seq_pred = SO_v[n, :]
        if np.sum(seq_pred) != 0:
            x = extract_event(seq_pred, n)
            event_p_v[n] = x
        seq_gt = GT_v[n, :]
        if np.sum(seq_gt) != 0:
            x = extract_event(seq_gt, n)
            event_gt_v[n] = x

        seq_pred = SO_av[n, :]
        if np.sum(seq_pred) != 0:
            x = extract_event(seq_pred, n)
            event_p_av[n] = x
        seq_gt = GT_av[n, :]
        if np.sum(seq_gt) != 0:
            x = extract_event(seq_gt, n)
            event_gt_av[n] = x

        tp, fp, fn = event_wise_metric(event_p_a[n], event_gt_a[n])
        TP_a[n] += tp
        FP_a[n] += fp
        FN_a[n] += fn

        tp, fp, fn = event_wise_metric(event_p_v[n], event_gt_v[n])
        TP_v[n] += tp
        FP_v[n] += fp
        FN_v[n] += fn

        tp, fp, fn = event_wise_metric(event_p_av[n], event_gt_av[n])
        TP_av[n] += tp
        FP_av[n] += fp
        FN_av[n] += fn

    TP = TP_a + TP_v
    FN = FN_a + FN_v
    FP = FP_a + FP_v

    n = len(FP_a)

    F_a = [None for _ in range(25)]
    F_v = [None for _ in range(25)]
    F = [None for _ in range(25)]
    F_av = [None for _ in range(25)]

    for ii in range(n):
        if (TP_a + FP_a)[ii] != 0 or (TP_a + FN_a)[ii] != 0:
            F_a[ii] = 2 * TP_a[ii] / (2 * TP_a[ii] + (FN_a + FP_a)[ii])

    for ii in range(n):
        if (TP_v + FP_v)[ii] != 0 or (TP_v + FN_v)[ii] != 0:
            F_v[ii] = 2 * TP_v[ii] / (2 * TP_v[ii] + (FN_v + FP_v)[ii])

    for ii in range(n):
        if (TP + FP)[ii] != 0 or (TP + FN)[ii] != 0:
            F[ii] = 2 * TP[ii] / (2 * TP[ii] + (FN + FP)[ii])

    for ii in range(n):
        if (TP_av + FP_av)[ii] != 0 or (TP_av + FN_av)[ii] != 0:
            F_av[ii] = 2 * TP_av[ii] / (2 * TP_av[ii] + (FN_av + FP_av)[ii])

    return F_a, F_v, F, F_av


def segment_level_by_cat(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av):
    # compute F scores
    # audio
    TP_a = np.sum(SO_a * GT_a, axis=1)  # 25
    FN_a = np.sum((1 - SO_a) * GT_a, axis=1)  # 25
    FP_a = np.sum(SO_a * (1 - GT_a), axis=1)  # 25

    F_a = [None for _ in range(25)]
    F_v = [None for _ in range(25)]
    F = [None for _ in range(25)]
    F_av = [None for _ in range(25)]

    n = len(FP_a)
    for ii in range(n):  # 对于每个类别计算F-score
        if (TP_a + FP_a)[ii] != 0 or (TP_a + FN_a)[ii] != 0:
            F_a[ii] = 2 * TP_a[ii] / (2 * TP_a[ii] + (FN_a + FP_a)[ii])

    # visual
    TP_v = np.sum(SO_v * GT_v, axis=1)
    FN_v = np.sum((1 - SO_v) * GT_v, axis=1)
    FP_v = np.sum(SO_v * (1 - GT_v), axis=1)

    for ii in range(n):
        if (TP_v + FP_v)[ii] != 0 or (TP_v + FN_v)[ii] != 0:
            F_v[ii] = 2 * TP_v[ii] / (2 * TP_v[ii] + (FN_v + FP_v)[ii])

    # audio-visual, 并集
    TP = TP_a + TP_v
    FN = FN_a + FN_v
    FP = FP_a + FP_v

    n = len(FP)

    for ii in range(n):
        if (TP + FP)[ii] != 0 or (TP + FN)[ii] != 0:
            F[ii] = 2 * TP[ii] / (2 * TP[ii] + (FN + FP)[ii])

    # audio-visual, 交集
    TP_av = np.sum(SO_av * GT_av, axis=1)
    FN_av = np.sum((1 - SO_av) * GT_av, axis=1)
    FP_av = np.sum(SO_av * (1 - GT_av), axis=1)
    n = len(FP_av)

    for ii in range(n):
        if (TP_av + FP_av)[ii] != 0 or (TP_av + FN_av)[ii] != 0:
            F_av[ii] = 2 * TP_av[ii] / (2 * TP_av[ii] + (FN_av + FP_av)[ii])

    return F_a, F_v, F, F_av


def count_confusion(pred, label):
    # pred.shape = (25,)
    matrix = np.zeros((26, 26))
    pidxes = pred != 0
    lidxes = label != 0

    if len(pidxes) != 0 and len(lidxes) != 0:
        for i in pidxes:
            for j in lidxes:
                matrix[i][j] += 1
    elif len(pidxes) != 0:
        for i in pidxes:
            matrix[i][25] += 1
    elif len(lidxes) != 0:
        for j in lidxes:
            matrix[25][j] += 1
    return matrix


def confusion_matrix(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av):
    # SO_a.shape = [25, 10]
    audio_matrix = np.zeros((26, 26))
    visual_matrix = np.zeros((26, 26))
    for i in range(10):
        audio_matrix += count_confusion(SO_a[:, i] + SO_av[:, i], GT_a[:, i] + GT_av[:, i])
        visual_matrix += count_confusion(SO_v[:, i] + SO_av[:, i], GT_v[:, i] + GT_av[:, i])
    return audio_matrix, visual_matrix


def to_vec(start, end):
    x = np.zeros(10)
    for i in range(start, end):
        x[i] = 1
    return x


def extract_event(seq, n):
    x = []
    i = 0
    while i < 10:
        if seq[i] == 1:
            start = i
            if i + 1 == 10:
                i = i + 1
                end = i
                x.append(to_vec(start, end))
                break

            for j in range(i + 1, 10):
                if seq[j] != 1:
                    i = j + 1
                    end = j
                    x.append(to_vec(start, end))
                    break
                else:
                    i = j + 1
                    if i == 10:
                        end = i
                        x.append(to_vec(start, end))
                        break
        else:
            i += 1
    return x


def event_wise_metric(event_p, event_gt):
    TP = 0
    FP = 0
    FN = 0

    if event_p is not None:  # 因为阈值是0.5,并且两个相邻的segment会被merge,因此不会产生两个pred对上同一个gt的情况
        num_event = len(event_p)
        for i in range(num_event):
            x1 = event_p[i]
            if event_gt is not None:
                nn = len(event_gt)
                flag = True
                for j in range(nn):
                    x2 = event_gt[j]
                    if np.sum(x1 * x2) >= 0.5 * np.sum(x1 + x2 - x1 * x2):  # 0.5
                        TP += 1
                        flag = False
                        break
                if flag:
                    FP += 1
            else:
                FP += 1

    if event_gt is not None:
        num_event = len(event_gt)
        for i in range(num_event):
            x1 = event_gt[i]
            if event_p is not None:
                nn = len(event_p)
                flag = True
                for j in range(nn):
                    x2 = event_p[j]
                    if np.sum(x1 * x2) >= 0.5 * np.sum(x1 + x2 - x1 * x2):  # 0.5
                        flag = False
                        break
                if flag:
                    FN += 1
            else:
                FN += 1
    return TP, FP, FN
