import numpy as np
import pandas as pd


def check_low_confidence_resp(F_seg_a, F_seg_v, F_seg, F_seg_av, F_event_a, F_event_v, F_event, F_event_av, val_loader, percent, file, anno_file):
    # 找到准确率最低的percent%的视频,写入文件
    F_seg_a = np.array(F_seg_a)
    F_seg_v = np.array(F_seg_v)
    F_seg = np.array(F_seg)
    F_seg_av = np.array(F_seg_av)

    F_event_a = np.array(F_event_a)
    F_event_v = np.array(F_event_v)
    F_event = np.array(F_event)
    F_event_av = np.array(F_event_av)

    number = int((percent / 100.) * len(val_loader))

    seg_a_idx = np.argsort(F_seg_a)[:number]
    seg_v_idx = np.argsort(F_seg_v)[:number]
    seg_idx = np.argsort(F_seg)[:number]
    seg_av_idx = np.argsort(F_seg_av)[:number]

    event_a_idx = np.argsort(F_event_a)[:number]
    event_v_idx = np.argsort(F_event_v)[:number]
    event_idx = np.argsort(F_event)[:number]
    event_av_idx = np.argsort(F_event_av)[:number]

    df = pd.read_csv(anno_file, header=0, sep='\t')
    videos = df.loc[:, :]
    filenames = videos[:, 0]
    # labels = videos[:, -1]

    with open(file, 'w', encoding='utf-8') as f:
        f.write("seg_a_idx\tF_seg_a\n")  # segment_audio
        tmp = ["\t".join([filenames[i], str(round(s, 2))]) for i, s in zip(seg_a_idx, F_seg_a[seg_a_idx])]
        f.write("\n".join(tmp))
        f.write("\n")

        f.write("seg_v_idx\tF_seg_v\n")  # segment_visual
        tmp = ["\t".join([filenames[i], str(round(s, 2))]) for i, s in zip(seg_v_idx, F_seg_v[seg_v_idx])]
        f.write("\n".join(tmp))
        f.write("\n")

        f.write("seg_idx\tF_seg\n")  # segment
        tmp = ["\t".join([filenames[i], str(round(s, 2))]) for i, s in zip(seg_idx, F_seg[seg_idx])]
        f.write("\n".join(tmp))
        f.write("\n")

        f.write("seg_av_idx\tF_seg_av\n")  # segment_audio_visual
        tmp = ["\t".join([filenames[i], str(round(s, 2))]) for i, s in zip(seg_av_idx, F_seg_av[seg_av_idx])]
        f.write("\n".join(tmp))
        f.write("\n")

        f.write("event_a_idx\tF_event_a\n")  # event_audio
        tmp = ["\t".join([filenames[i], str(round(s, 2))]) for i, s in zip(event_a_idx, F_event_a[event_a_idx])]
        f.write("\n".join(tmp))
        f.write("\n")

        f.write("event_v_idx\tF_event_v\n")  # event_visual
        tmp = ["\t".join([filenames[i], str(round(s, 2))]) for i, s in zip(event_v_idx, F_event_v[event_v_idx])]
        f.write("\n".join(tmp))
        f.write("\n")

        f.write("event_idx\tF_event\n")  # event
        tmp = ["\t".join([filenames[i], str(round(s, 2))]) for i, s in zip(event_idx, F_event[event_idx])]
        f.write("\n".join(tmp))
        f.write("\n")

        f.write("event_av_idx\tF_event_av\n")  # event_audio_visual
        tmp = ["\t".join([filenames[i], str(round(s, 2))]) for i, s in zip(event_av_idx, F_event_av[event_av_idx])]
        f.write("\n".join(tmp))
        f.write("\n")


def check_low_confidence_toge(F_seg_a, F_seg_v, F_seg, F_seg_av, F_event_a, F_event_v, F_event, F_event_av, val_loader, percent, file, anno_file):
    F_seg_a = np.array(F_seg_a)
    F_seg_v = np.array(F_seg_v)
    F_seg = np.array(F_seg)
    F_seg_av = np.array(F_seg_av)

    F_event_a = np.array(F_event_a)
    F_event_v = np.array(F_event_v)
    F_event = np.array(F_event)
    F_event_av = np.array(F_event_av)

    scores = F_seg_a + F_seg_v + F_seg + F_seg_av + F_event_a + F_event_v + F_event + F_event_av
    number = int((percent / 100.) * len(val_loader))

    df = pd.read_csv(anno_file, header=0, sep='\t')
    videos = df.loc[:, :]
    filenames = videos[:, 0]

    idxes = np.argsort(scores)[:number]
    with open(file, 'w', encoding='utf-8') as f:
        f.write("idx\ttotal_score\n")
        tmp = ["\t".join([filenames[i], str(round(s, 2))]) for i, s in zip(idxes, scores[idxes])]
        f.write("\n".join(tmp))


def write_conf(F_seg_a, F_seg_v, F_seg, F_seg_av, F_event_a, F_event_v, F_event, F_event_av, file, anno_file):
    F_seg_a = np.array(F_seg_a)
    F_seg_v = np.array(F_seg_v)
    F_seg = np.array(F_seg)
    F_seg_av = np.array(F_seg_av)
    F_event_a = np.array(F_event_a)
    F_event_v = np.array(F_event_v)
    F_event = np.array(F_event)
    F_event_av = np.array(F_event_av)

    scores = F_seg_a + F_seg_v + F_seg + F_seg_av + F_event_a + F_event_v + F_event + F_event_av
    idxes = np.argsort(scores)

    df = pd.read_csv(anno_file, header=0, sep='\t')
    filenames = df["filename"]
    # filenames = videos[:, 0]

    with open(file, 'w', encoding='utf-8') as f:
        f.write("filename\ttotal_score\n")
        tmp = ["\t".join([filenames[i], str(round(s, 2))]) for i, s in zip(idxes, scores[idxes])]
        f.write("\n".join(tmp))


def print_overall_metric(F_seg_a, F_seg_v, F_seg, F_seg_av, F_event_a, F_event_v, F_event, F_event_av):
    print('Audio Event Detection Segment-level F1: {:.1f}'.format(100 * np.mean(np.array(F_seg_a))))
    print('Visual Event Detection Segment-level F1: {:.1f}'.format(100 * np.mean(np.array(F_seg_v))))
    print('Audio-Visual Event Detection Segment-level F1: {:.1f}'.format(100 * np.mean(np.array(F_seg_av))))

    avg_type = (100 * np.mean(np.array(F_seg_av)) +
                100 * np.mean(np.array(F_seg_a)) +
                100 * np.mean(np.array(F_seg_v))) / 3.
    avg_event = 100 * np.mean(np.array(F_seg))
    print('Segment-level Type@Avg. F1: {:.1f}'.format(avg_type))
    print('Segment-level Event@Avg. F1: {:.1f}'.format(avg_event))

    print('Audio Event Detection Event-level F1: {:.1f}'.format(100 * np.mean(np.array(F_event_a))))
    print('Visual Event Detection Event-level F1: {:.1f}'.format(100 * np.mean(np.array(F_event_v))))
    print('Audio-Visual Event Detection Event-level F1: {:.1f}'.format(100 * np.mean(np.array(F_event_av))))

    avg_type_event = (100 * np.mean(np.array(F_event_av)) +
                      100 * np.mean(np.array(F_event_a)) +
                      100 * np.mean(np.array(F_event_v))) / 3.
    avg_event_level = 100 * np.mean(np.array(F_event))
    print('Event-level Type@Avg. F1: {:.1f}'.format(avg_type_event))
    print('Event-level Event@Avg. F1: {:.1f}'.format(avg_event_level))

    return avg_type


def log_metric_by_cat(F_seg_a, F_seg_v, F_seg, F_seg_av, F_event_a, F_event_v, F_event, F_event_av, file):
    F_seg_a = np.array(F_seg_a)  # [number, 25]
    F_seg_v = np.array(F_seg_v)
    F_seg = np.array(F_seg)
    F_seg_av = np.array(F_seg_av)
    F_event_a = np.array(F_event_a)
    F_event_v = np.array(F_event_v)
    F_event = np.array(F_event)
    F_event_av = np.array(F_event_av)

    seg_a = np.zeros(25)
    seg_v = np.zeros(25)
    seg = np.zeros(25)
    seg_av = np.zeros(25)
    event_a = np.zeros(25)
    event_v = np.zeros(25)
    event = np.zeros(25)
    event_av = np.zeros(25)

    for i in range(25):
        idxes = F_seg_a[:, i] != None
        seg_a[i] = sum(F_seg_a[:, i][idxes]) / sum(idxes)

        idxes = F_seg_v[:, i] != None
        seg_v[i] = sum(F_seg_v[:, i][idxes]) / sum(idxes)

        idxes = F_seg[:, i] != None
        seg[i] = sum(F_seg[:, i][idxes]) / sum(idxes)

        idxes = F_seg_av[:, i] != None
        seg_av[i] = sum(F_seg_av[:, i][idxes]) / sum(idxes)

        idxes = F_event_a[:, i] != None
        event_a[i] = sum(F_event_a[:, i][idxes]) / sum(idxes)

        idxes = F_event_v[:, i] != None
        event_v[i] = sum(F_event_v[:, i][idxes]) / sum(idxes)

        idxes = F_event[:, i] != None
        event[i] = sum(F_event[:, i][idxes]) / sum(idxes)

        idxes = F_event_av[:, i] != None
        event_av[i] = sum(F_event_av[:, i][idxes]) / sum(idxes)

    scores = (seg_a + seg_v + seg + seg_av + event_a + event_v + event + event_av) / 8.
    categories = np.array(['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                  'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                  'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                  'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                  'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                  'Clapping'])
    idxes = np.argsort(scores)
    cat_score = ["\t".join([categories[i], str(round(scores[i], 2))]) for i in idxes]
    with open(file, 'w', encoding='utf-8') as f:
        f.write("\n".join(cat_score))

    return scores