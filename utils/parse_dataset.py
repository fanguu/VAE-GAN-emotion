import os
import csv
import copy
import numpy as np
from random import shuffle
import shutil
import string



def get_dataset_files(data_dir, data_ext, celeb_ids):
    """
    从文件夹中读取voice或face数据
    """
    new_folder_root = '/home/fz/2-VF-feature/JVF-net/dataset/voclexb-VGG_face-datasets/1-vox1-VGG'
    data_list = []
    # # rename image folder(face) name to ID, e.g. Admir_kHan --> id10001
    # for root, dirs, _ in sorted(os.walk(data_dir)):
    #     for dir in dirs:
    #         folder_pth = os.path.join(root, dir)
    #         folder = folder_pth[len(data_dir):].split('/')[1]
    #         celeb_name = celeb_ids.get(folder, folder)    # default_value不设置的话默认为None，设置的话即如果找不到则返回default设定的值
    #         if celeb_name != folder:
    #             new_folder_pth = os.path.join(new_folder_root, celeb_name)
    #             shutil.copytree(folder_pth,new_folder_pth)
    #             # os.rename(folder_pth, new_folder_pth)

    # read data directory to create data_list
    for root, dirs, filenames in sorted(os.walk(data_dir)):
        for filename in filenames:
            if filename.endswith(data_ext):
                filepath = os.path.join(root, filename)
                folder = filepath[len(data_dir):].split('/')[1]
                celeb_name = celeb_ids.get(folder, folder) #  default_value不设置的话默认为None，设置的话即如果找不到则返回default设定的值
                if celeb_name != folder and data_ext =='jpg':
                    data_list.append({'folder_id':folder, 'name': celeb_name, 'face_path': filepath})
                if celeb_name != folder and data_ext =='wav':
                    data_list.append({'folder_id':folder, 'name': celeb_name, 'voice_path': filepath})


    return data_list


def get_voclexb_labels(voice_list, face_list, celeb_ids):
    """
    合并voice和face中的同类项目
    """
    voice_names = {item['name'] for item in voice_list}
    face_names = {item['name'] for item in face_list}
    names = voice_names & face_names
    voice_face_list = []
    label_dict = {}

    #  通过列表推导式 保留同类项
    voice_list = [item for item in voice_list if item['name'] in names]
    face_list = [item for item in face_list if item['name'] in names]

    names = list(sorted(names))      # 增加排序, 固定名字与序列号
    for step, item in enumerate(names):
        label_dict[item] = step
    label_dict = dict(zip(names, range(len(names))))


    # 建立face-list,
    for item in voice_list+face_list:
        item['actor_id'] = label_dict[item['name']]


    return voice_list, face_list


def get_voclexb_csv(csv_files, voice_folder, face_folder):
    """
    从list.csv中读取路径, 写入list中,
    :param data_params:
    :return: 数据路径以及标签,speaker数量
    """
    csv_voice_headers = ['folder_id', 'actor_id', 'name', 'voice_path']
    csv_face_headers = ['folder_id',  'actor_id', 'name', 'face_path']
    triplet_list = []
    actor_dict, actor_dict1 = {}, {}

    with open(csv_files) as f:
        lines = f.readlines()[1:]
        for line in lines:
            actor_ID, name, gender, nation, _ = line.rstrip("\n").split('\t')
            actor_dict[actor_ID] = name
            actor_dict1[name] = actor_ID

    face_list=[]

    face_list = get_dataset_files(face_folder, 'jpg', actor_dict)
    voice_list = get_dataset_files(voice_folder, 'wav', actor_dict)
    voice_list, face_list = get_voclexb_labels(voice_list, face_list, actor_dict1)

    csv_face_pth = os.path.join('../dataset/voclexb-VGG_face-datasets/', 'vocelexb_face.csv')
    print(csv_face_pth)
    with open(csv_face_pth,'w',newline='', ) as f:
        f_scv = csv.DictWriter(f, csv_face_headers, delimiter = ',', lineterminator = '\n')
        f_scv.writeheader()
        f_scv.writerows(face_list)

    csv_voice_pth = os.path.join('../dataset/voclexb-VGG_face-datasets/', 'vocelexb_voice.csv')
    print(csv_voice_pth)
    with open(csv_voice_pth,'w',newline='', ) as f:
        f_scv = csv.DictWriter(f, csv_voice_headers, delimiter = ',', lineterminator = '\n')
        f_scv.writeheader()
        f_scv.writerows(voice_list)

    return len(actor_dict)


def get_labels(dataset='ucf101'):
    labels = []
    with open('/home/fz/1-Dataset/preprocessed_ucf101/ucf101_labels.txt', 'r') as load_f:
        raw_labels = load_f.readlines()
    for label in raw_labels:
        labels.append(label.replace('\n', ''))
    return sorted(labels)


def get_ucf101_csv():
    preprocessed_dir = '/home/fz/1-Dataset/preprocessed_ucf101'
    split = 'val'
    data_list = []
    file_names, labels = [], []
    for label in sorted(os.listdir(os.path.join(preprocessed_dir, split))):
        for file_name in sorted(os.listdir(os.path.join(preprocessed_dir, split, label))):
            file_names.append(os.path.join(preprocessed_dir, split, label, file_name))
            labels.append(label)

    print('Number of {} videos: {:d}'.format(split, len(file_names)))

    # prepare a mapping between the label names (strings) and indices (ints)
    label2index = {label: index for index, label in enumerate(get_labels('ucf101'))}
    # convert the list of label names into an array of label indices
    label_array = [label2index[label] for label in labels]

    data_list = []
    for i, action in enumerate(label_array):

        headers = ['actor_id', 'gender', 'vocal_channel', 'emotion', 'emotion_intensity', 'voice_path']
        data_list.append({'actor_id': 0, 'gender': 0, 'vocal_channel': 0,
                          'emotion': action, 'emotion_intensity': 0, 'voice_path': file_names[i]})

    print("sample numbers:{}".format(len(data_list)))

    csv_pth = os.path.join(preprocessed_dir, 'ucf101_{}.csv'.format(split))
    print("csv_pth:{}".format(csv_pth))
    with open(csv_pth, 'w', newline='') as f:
        f_scv = csv.DictWriter(f, headers)
        f_scv.writeheader()
        f_scv.writerows(data_list)



def get_RAVDESS_voice_csv(data_pth, csv_pth, data_ext):
    """
    从音频特征或图像文件夹中读取对应文件, 在csv中写入该文件路径,情感,身份,性别标签
    :param image_data_pth: 图像文件抽取中间为代表图像
    :param csv_pth: csv文件输出位置
    :param data_ext: .npy或者.png格式
    :return:
    """
    data_list = []
    list_name ={"image":"png", "voice":"wav", "mfcc":"npy", "fbank":"npy", "spectrogram":"spectrogram"}

    file_ext = list_name[data_ext]
    headers = ['actor_id', 'gender', 'vocal_channel', 'emotion', 'emotion_intensity', 'voice_path']
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    # read data directory
    for root, folders, filenames in os.walk(data_pth):      # 音频数据集根目录, 子目录, 文件名
        folders.sort()
        filenames.sort()
        for filename in filenames:
            if filename.endswith(file_ext):              # 校验文件后缀名, wav或者npy
                voice_path = os.path.join(root, filename)
                flag = filename.split('.')[0].split('-')
                if flag[0] == '01':  # only use video
                    gend = "female" if int(flag[6])%2 else "male"
                    data_list.append({'actor_id':flag[6], 'gender':gend, 'vocal_channel':flag[1],
                                      'emotion':flag[2], 'emotion_intensity':flag[3], 'voice_path': voice_path})
                    print("voice_{0:}_path:{1:}, actor:{2:}".format(data_ext, voice_path, flag[6]))
                else:
                    print('not video')

    print("sample numbers:{}".format(len(data_list)))

    csv_pth = os.path.join(csv_pth, 'RAVDESS_{}.csv'.format('mfcc'))
    print("csv_pth:{}".format(csv_pth))
    with open(csv_pth,'w',newline='') as f:
        f_scv = csv.DictWriter(f,headers)
        f_scv.writeheader()
        f_scv.writerows(data_list)


def get_RAVDESS_video_csv(data_pth, csv_pth, data_ext, val_ratio=0.2):
    """
    从音频特征或图像文件夹中读取对应文件, 在csv中写入该文件路径,情感,身份,性别标签, 并分为训练集、验证集
    :param image_data_pth: 图像文件抽取中间为代表图像
    :param csv_pth: csv文件输出位置
    :param data_ext: .npy或者.png格式
    :return:
    """
    data_list = []
    train_list = []
    test_list = []
    list_name ={"image":"png", "voice":"wav"}

    file_ext = list_name[data_ext]
    headers = ['actor_id','gender','vocal_channel','emotion','emotion_intensity', 'video_path']
    emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    # read data directory
    for filenames in sorted(os.listdir(data_pth)):      # 音频数据集根目录, 子目录, 文件名
        for videonames in sorted(os.listdir(os.path.join(data_pth, filenames))):
            video_path = os.path.join(data_pth, filenames, videonames)
            flag = videonames.split('-')
            if flag[0] == '01':  # only use video
                gend = "female" if int(flag[6])%2 else "male"
                data_list.append({'actor_id':flag[6], 'gender':gend, 'vocal_channel':flag[1],
                                  'emotion':flag[2], 'emotion_intensity':flag[3], 'video_path': video_path})
                print("face_{0:}_path:{1:}, actor:{2:}".format(data_ext, video_path, flag[6]))

    print("sample numbers:{}".format(len(data_list)))

    shuffle(data_list)
    N_test_list = int(len(data_list[:]) * val_ratio)

    train_list = data_list[N_test_list:]
    test_list = data_list[:N_test_list]
    train_list = sorted(train_list, key=lambda i: (i['actor_id'], i['emotion']))
    test_list = sorted(test_list, key=lambda i: (i['actor_id'], i['emotion']))

    train_csv = os.path.join(csv_pth, 'RAVDESS_video_train.csv')
    print("csv_pth:{}".format(train_csv))
    with open(train_csv,'w',newline='') as f:
        f_scv = csv.DictWriter(f, headers)
        f_scv.writeheader()
        f_scv.writerows(train_list)

    test_csv = os.path.join(csv_pth, 'RAVDESS_video_test.csv')
    print("csv_pth:{}".format(test_csv))
    with open(test_csv,'w',newline='') as f:
        f_scv = csv.DictWriter(f, headers)
        f_scv.writeheader()
        f_scv.writerows(test_list)



def RAVDESS_csv_to_list():
    pass


if __name__ == '__main__':
    # get_RAVDESS_dataset(DATASET_PARAMETERS)
    # data_dir = 'data/RAVDESS/fbank'

    # csv_files = '/home/fz/2-VF-feature/JVF-net/dataset/voclexb-VGG_face-datasets/vox1_meta.csv'
    # voice_folder = '/home/fz/2-VF-feature/JVF-net/dataset/voclexb-VGG_face-datasets/1-vox1-wav-all'
    # face_folder = '/home/fz/2-VF-feature/JVF-net/dataset/voclexb-VGG_face-datasets/0-vox1-VGG-face'
    # num = get_voclexb_csv(csv_files, voice_folder, face_folder)

    voice_data_pth = '/home/fz/3-semi-supervised-emotion/VAE-GAN-emotion/dataset/RAVDESS/2 mfcc-Actor1-24-16k'
    image_data_pth = '/home/fz/1-Dataset/RAVDESS/1 image-Actor1-24-freq4-crop'
    csv_pth = "/home/fz/2-VF-feature/JVF-net/dataset/RAVDESS"
    get_RAVDESS_video_csv( image_data_pth, csv_pth, 'image')
    # get_RAVDESS_voice_csv( voice_data_pth, csv_pth, 'mfcc')


    # get_ucf101_csv()

