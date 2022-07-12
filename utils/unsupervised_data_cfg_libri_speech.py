import json
#import librosa
import argparse
import random
from random import shuffle
import numpy as np
import torchaudio
import os

def get_file_dur(fname):
    try:
        x, rate = torchaudio.load(fname)
    except RuntimeError:
        print("Error processing {fname}")
        return (0)

    return x.shape[1]


def main(opts):
    random.seed(opts.seed)
    # spk2idx = np.load(opts.libri_dict, allow_pickle=True)  # 保存对象数组
    # spk2idx = dict(spk2idx.any())
    spk2idx = {}

    data_cfg = {'train':{'data':[],
                         'speakers':[]},
                'valid':{'data':[],
                         'speakers':[]},
                'test':{'data':[],
                        'speakers':[]},
                'speakers':[],
                'emotions':[]}
    with open(opts.train_scp, 'r') as train_f:
        train_files = train_f.readlines()[1:]
        # train_files = [l.rstrip() for l in train_f]
        shuffle(train_files)    # 随机排序
        if opts.valid_scp is None:
            N_valid_files = int(len(train_files) * opts.val_ratio)
            valid_files = train_files[:N_valid_files]
            train_files = train_files[N_valid_files:]

        train_dur = 0
        print('Processing train file {:d}'.format(len(train_files)))
        for ti, train_file in enumerate(train_files, start=1):
            actor_id, gender, vocal_channel, emotion, emotion_intensity, wave_path = train_file.rstrip("\n").split(',')
            # actor_ID, emotion, wave_path = train_file.rstrip("\n").split(',')
            if actor_id not in data_cfg['speakers']:
                data_cfg['speakers'].append(actor_id)
            if emotion not in data_cfg['emotions']:
                data_cfg['emotions'].append(emotion)
            # spk = spk2idx[wave_path]  # 读取example_librispeech_dict.npy数据
            # data_cfg['speakers'].append(actor_ID)
            # data_cfg['train']['speakers'].append(actor_ID)
            data_cfg['train']['data'].append({'filename': wave_path,
                                              'speaker': actor_id,
                                              'emotion': emotion
                                              })
            train_dur += get_file_dur(wave_path)
            # train_dur = 0
        data_cfg['train']['total_wav_dur'] = train_dur
        print('train audio during times {:d}'.format(train_dur))

        # 从训练集中抽取的验证集
        print('Processing valid file {:d}'.format(len(valid_files))) # 可利用 end='\r' 换行输出
        if opts.valid_scp is None:
            valid_dur = 0
            for ti, valid_file in enumerate(valid_files, start=1):
                actor_id, gender, vocal_channel, emotion, emotion_intensity, wave_path = valid_file.rstrip("\n").split(',')
                if actor_id not in data_cfg['speakers']:
                    data_cfg['speakers'].append(actor_id)
                if emotion not in data_cfg['emotions']:
                    data_cfg['emotions'].append(emotion)
                # data_cfg['valid']['speakers'].append(actor_ID)
                data_cfg['valid']['data'].append({'filename':wave_path,
                                                  'speaker':actor_id,
                                                  'emotion': emotion
                                                  })
                # valid_dur += get_file_dur(wave_path)
                valid_dur = 0
            data_cfg['valid']['total_wav_dur'] = valid_dur
            print('test audio during times {:d}'.format(valid_dur))

    # 创建专门的验证集
    if opts.valid_scp is not None:
        with open(opts.valid_scp, 'r') as valid_f:
            valid_files = [l.rstrip() for l in valid_f]
            valid_dur = 0
            for ti, valid_file in enumerate(valid_files, start=1):
                print('Processing valid file {:7d}/{:7d}'.format(ti,
                                                                 len(valid_files)),
                      end='\r')
                spk = spk2idx[valid_file]
                if spk not in data_cfg['speakers']:
                    data_cfg['speakers'].append(spk)
                    data_cfg['valid']['speakers'].append(spk)
                data_cfg['valid']['data'].append({'filename':valid_file,
                                                  'spk':spk})
                valid_dur += get_file_dur(os.path.join(opts.data_root,
                                                       valid_file))
            data_cfg['valid']['total_wav_dur'] = valid_dur
            print()

    # 创建专门的测试集
    if opts.test_scp is not None:
        with open(opts.test_scp, 'r') as test_f:
            test_files = [l.rstrip() for l in test_f]
            test_dur = 0
            for ti, test_file in enumerate(test_files, start=1):
                print('Processing test file {:7d}/{:7d}'.format(ti, len(test_files)), end='\r')
                spk = spk2idx[test_file]
                if spk not in data_cfg['speakers']:
                    data_cfg['speakers'].append(spk)
                    data_cfg['test']['speakers'].append(spk)
                data_cfg['test']['data'].append({'filename':test_file,
                                                  'spk':spk})
                test_dur += get_file_dur(os.path.join(opts.data_root,
                                                      test_file))
            data_cfg['test']['total_wav_dur'] = test_dur

    with open(opts.cfg_file, 'w') as cfg_f:
        cfg_f.write(json.dumps(data_cfg))  # python 对象转化为 json
        print(opts.cfg_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='dataset/eNTERFACE')   # 1
    parser.add_argument('--train_scp', type=str, default='dataset/eNTERFACE/eNTERFACE_wav.csv')  # 2
    parser.add_argument('--valid_scp', type=str, default=None)
    parser.add_argument('--test_scp', type=str, default=None)  # 3
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio to take out of training'
                             'in utterances ratio (Def: 0.1).')
    parser.add_argument('--cfg_file', type=str, default='dataset/eNTERFACE/eNTERFACE_voice_data.cfg')
    parser.add_argument('--libri_dict', type=str, default='dataset/LibriSpeech/libri_dict.npy')  # 4
    parser.add_argument('--seed', type=int, default=3)
    
    opts = parser.parse_args()
    main(opts)

