import math
import os
import json
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# global configs
CLIP_LEN, RESIZE_HEIGHT, CROP_SIZE = 16, 128, 112


video_transform_train = transforms.Compose([
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                      transforms.Resize(size=RESIZE_HEIGHT),
                                      transforms.RandomCrop(size =CROP_SIZE ),
                                      transforms.RandomHorizontalFlip(p=0.5)
                                     ])



class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.
        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
    """

    def __init__(self, data_csv, split='train', verbose =True):

        self.data_csv = data_csv

        self.split = split

        headers = ['actor_id', 'gender', 'vocal_channel', 'emotion', 'emotion_intensity', 'video_path']

        self.actor_num, self.emotion_num, self.intensity_num = [], [], []
        self.data_list = []
        with open(self.data_csv, 'r') as f_csv:
            lines = f_csv.readlines()[0:]
            headers = lines[0]
            train_files = lines[1:]
            for line in train_files[:]:
                actor_id, gender, vocal_channel, emotion, emotion_intensity, video_path = line.rstrip("\n").split(',')
                self.actor_num.append(int(actor_id))
                self.emotion_num.append(int(emotion))
                self.intensity_num.append(int(emotion_intensity))
                # voice_list.append({'filepath': wave_path, 'name_id': actor_ID, 'emotion_id': emotion})
                self.data_list.append({'video_path': video_path, 'actor_id': actor_id, 'emotion': emotion, 'emotion_intensity':emotion_intensity})

        print('Number of {} videos: {:d};'.format(split, len(self.data_list)))


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # load and preprocess.
        video_path = self.data_list[index]['video_path']
        emotion_label = int(self.data_list[index]['emotion'])-1
        emotion_label = np.array(emotion_label)
        buffer = self.load_frames(video_path)

        buffer = self.crop(buffer, CLIP_LEN, CROP_SIZE)
        if self.split == 'train':
            # perform data augmentation (random horizontal flip)
            buffer = self.random_flip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)

        return torch.from_numpy(buffer), torch.from_numpy(emotion_label)

    def check_integrity(self):
        if os.path.exists(os.path.join(self.original_dir, self.split)):
            return True
        else:
            return False


    def preprocess(self):
        if not os.path.exists(self.preprocessed_dir):
            os.mkdir(self.preprocessed_dir)
        os.mkdir(os.path.join(self.preprocessed_dir, self.split))

        for file in sorted(os.listdir(os.path.join(self.original_dir, self.split))):
            os.mkdir(os.path.join(self.preprocessed_dir, self.split, file))

            for video in sorted(os.listdir(os.path.join(self.original_dir, self.split, file))):
                video_name = os.path.join(self.original_dir, self.split, file, video)
                save_name = os.path.join(self.preprocessed_dir, self.split, file, video)
                self.process_video(video_name, save_name)

        print('Preprocess finished.')

    @staticmethod
    def process_video(video_name, save_name):
        print('Preprocess {}'.format(video_name))
        # initialize a VideoCapture object to read video data into a numpy array
        capture = cv2.VideoCapture(video_name)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

        # make sure the preprocessed video has at least CLIP_LEN frames
        extract_frequency = 4
        if frame_count // extract_frequency <= CLIP_LEN:
            extract_frequency -= 1
            if frame_count // extract_frequency <= CLIP_LEN:
                extract_frequency -= 1
                if frame_count // extract_frequency <= CLIP_LEN:
                    extract_frequency -= 1

        count, i, retaining = 0, 0, True
        while count < frame_count and retaining:
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % extract_frequency == 0:
                resize_height = RESIZE_HEIGHT
                resize_width = math.floor(frame_width / frame_height * resize_height)
                # make sure resize width >= crop size
                if resize_width < CROP_SIZE:
                    resize_width = RESIZE_HEIGHT
                    resize_height = math.floor(frame_height / frame_width * resize_width)

                frame = cv2.resize(frame, (resize_width, resize_height))
                if not os.path.exists(save_name.split('.')[0]):
                    os.mkdir(save_name.split('.')[0])
                cv2.imwrite(filename=os.path.join(save_name.split('.')[0], '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # release the VideoCapture once it is no longer needed
        capture.release()

    @staticmethod
    def random_flip(buffer):
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = frame

        return buffer

    def normalize(self, buffer):
        buffer = buffer.astype(np.float32)
        for i, frame in enumerate(buffer):
            frame = frame / 255.0
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        buffer = []
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name))
            frame = video_transform_train(frame)
            # width, height, number = frame.shape
            # resize_height = RESIZE_HEIGHT
            # resize_width = math.floor(width / height * resize_height)
            # # make sure resize width >= crop size
            # if resize_width < CROP_SIZE:
            #     resize_width = RESIZE_HEIGHT
            #     resize_height = math.floor(height / width * resize_width)
            #
            # frame = cv2.resize(frame, (resize_width, resize_height))
            buffer.append(frame)

        return np.array(buffer).astype(np.uint8)

    def crop(self, buffer, clip_len, crop_size):
        if self.split == 'train':
            # randomly select time index for temporal jitter
            if buffer.shape[0] > clip_len:
                time_index = np.random.randint(buffer.shape[0] - clip_len)
            else:
                time_index = 0
            # randomly select start indices in order to crop the video
            height_index = np.random.randint(buffer.shape[1] - crop_size)
            width_index = np.random.randint(buffer.shape[2] - crop_size)
            # crop and jitter the video using indexing. The spatial crop is performed on
            # the entire array, so each frame is cropped in the same location. The temporal
            # jitter takes place via the selection of consecutive frames
        else:
            # for val and test, select the middle and center frames
            if buffer.shape[0] > clip_len:
                time_index = math.floor((buffer.shape[0] - clip_len) / 2)
            else:
                time_index = 0
            height_index = math.floor((buffer.shape[1] - crop_size) / 2)
            width_index = math.floor((buffer.shape[2] - crop_size) / 2)
        buffer = buffer[time_index:time_index + clip_len, height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        # padding repeated frames to make sure the shape as same
        if buffer.shape[0] < clip_len:
            repeated = clip_len // buffer.shape[0] - 1
            remainder = clip_len % buffer.shape[0]
            buffered, reverse = buffer, True
            if repeated > 0:
                padded = []
                for i in range(repeated):
                    if reverse:
                        pad = buffer[::-1, :, :, :]
                        reverse = False
                    else:
                        pad = buffer
                        reverse = True
                    padded.append(pad)
                padded = np.concatenate(padded, axis=0)
                buffer = np.concatenate((buffer, padded), axis=0)
            if reverse:
                pad = buffered[::-1, :, :, :][:remainder, :, :, :]
            else:
                pad = buffered[:remainder, :, :, :]
            buffer = np.concatenate((buffer, pad), axis=0)

        return buffer

    def pading(self, buffer, clip_len):
        # padding repeated frames to make sure the shape as same
        if buffer.shape[0] < clip_len:
            repeated = clip_len // buffer.shape[0] - 1
            remainder = clip_len % buffer.shape[0]
            buffered, reverse = buffer, True
            if repeated > 0:
                padded = []
                for i in range(repeated):
                    if reverse:
                        pad = buffer[::-1, :, :, :]
                        reverse = False
                    else:
                        pad = buffer
                        reverse = True
                    padded.append(pad)
                padded = np.concatenate(padded, axis=0)
                buffer = np.concatenate((buffer, padded), axis=0)
            if reverse:
                pad = buffered[::-1, :, :, :][:remainder, :, :, :]
            else:
                pad = buffered[:remainder, :, :, :]
            buffer = np.concatenate((buffer, pad), axis=0)
        return buffer


def load_data(train_csv, test_csv, batch_size=8):
    train_data = VideoDataset(data_csv=train_csv, split='train')
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16)
    # val_data = VideoDataset(dataset=dataset, split='val')
    # val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)
    test_data = VideoDataset(data_csv=test_csv, split='test')
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16)
    val_loader = test_loader
    return train_loader, val_loader, test_loader


def get_labels(dataset='ucf101'):
    labels = []
    with open('data/{}_labels.txt'.format(dataset), 'r') as load_f:
        raw_labels = load_f.readlines()
    for label in raw_labels:
        labels.append(label.replace('\n', ''))
    return sorted(labels)

class VGGDataset(Dataset):
    def __init__(self):
        pass

if __name__ == '__main__':

    train_data = VideoDataset(data_csv='/home/fz/3-semi-supervised-emotion/VAE-GAN-emotion/dataset/RAVDESS/RAVDESS_video_train.csv', split='train')
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=0)
    train_loader = iter(train_loader)
    a = next(train_loader)
    pass
    # data_root = '/home/fz/2-VF-feature/JVF-net/dataset/RAVDESS'
    # data_cfg = '/home/fz/2-VF-feature/JVF-net/dataset/RAVDESS/RAVDESS_image_data.cfg'
    # trans = None
    # face_dataset = RAVDESS_face_Dataset(data_root, data_cfg, split='valid')
    #
    # face_loader = DataLoader(face_dataset, batch_size=24, shuffle=True, drop_last=False, num_workers=0)
    # face_loader = iter(face_loader)
    # for i in range(10):
    #     data, label = next(face_loader)
    #     print(data.shape)  # (B, 1, 512, 300)
    #     print(label.shape)  # (B)
