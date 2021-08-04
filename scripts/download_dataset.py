import os
import os.path as osp
import pandas as pd
import time
from multiprocessing import Process
import paramiko


def download(path_data, name, t_seg):
    if not os.path.exists(path_data):
        os.makedirs(path_data)
    link_prefix = "https://www.youtube.com/watch?v="

    filename_full_video = os.path.join(path_data, name) + "_full_video.mp4"
    filename = os.path.join(path_data, name) + ".mp4"
    link = link_prefix + name

    if os.path.exists(filename):
        print("already exists, skip")
        return 0

    command1 = 'youtube-dl --ignore-config '
    command1 += link + " "
    command1 += "-o " + filename_full_video + " "
    command1 += "-f best "

    # command1 += '-q '  # print no log
    # print command1
    os.system(command1)

    t_start, t_end = t_seg
    t_dur = t_end - t_start
    print("trim the video to [%.1f-%.1f]" % (t_start, t_end))
    command2 = 'ffmpeg '
    command2 += '-ss '
    command2 += str(t_start) + ' '
    command2 += '-i '
    command2 += filename_full_video + ' '
    command2 += '-t '
    command2 += str(t_dur) + ' '
    command2 += '-vcodec libx264 '
    command2 += '-acodec aac -strict -2 '
    command2 += filename + ' '
    command2 += '-y '  # overwrite without asking
    command2 += '-loglevel -8 '  # print no log
    # print(command2)
    os.system(command2)
    try:
        os.remove(filename_full_video)
    except:
        return -1

    return 1


def upload(filename):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname="210.28.134.151", username="chenghaoyue", password="passwdCHY123", port=12022)
    sftp = ssh.open_sftp()
    for file in filename:
        localpath = "data/LLP_dataset/video/" + file
        remotepath = "/home/chenghaoyue/data2/projects/AVVP-ECCV20/data/LLP_dataset/video/" + file
        sftp.put(localpath, remotepath, callback=None)
    ssh.close()


def multithread_process(filenames):
    set = "data/LLP_dataset/video"
    length = len(filenames)
    stored_files = list()
    for i in range(length):
        row = filenames[i]
        name = row[:11]
        steps = row[11:].split("_")
        t_start = float(steps[1])
        t_end = t_start + 10
        state = download(set, name, (t_start, t_end))
        if state == 1 and osp.exists(osp.join(set, name) + ".mp4"):
            stored_files.append(name + ".mp4")
            if len(stored_files) == 100:
                upload(stored_files)
                stored_files = list()
    if len(stored_files) > 0:
        upload(stored_files)


if __name__ == '__main__':
    filename_source = "data/audioset_remained.csv"
    filenames = list(pd.read_csv(filename_source, header=0, sep='\t')["filename"].values)
    length = len(filenames)
    print(length)
    procs = 30
    length_per_process = (length + procs - 1) // procs
    start = time.time()
    proc_list = list()
    for i in range(procs):
        proc = Process(target=multithread_process,
                       args=(filenames[length_per_process * i: length_per_process * (i + 1)],))
        proc.start()
        proc_list.append(proc)
    for p in proc_list:
        p.join()
    end = time.time()
    print((end - start) / 3600.)
