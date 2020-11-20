import argparse
import json
from pathlib import Path

import pandas as pd

from utils import get_n_frames, get_n_frames_hdf5


def convert_csv_to_dict(csv_path, subset):
    data = pd.read_csv(csv_path)
    keys = []
    key_labels = []
    for i in range(data.shape[0]):
        row = data.iloc[i, :]
        basename = '%s' % (row['youtube_id'])
        keys.append(basename)
        if subset != 'testing':
            key_labels.append(row['label'])

    database = {}
    for i in range(len(keys)):
        key = keys[i]
        database[key] = {}
        database[key]['subset'] = subset
        if subset != 'testing':
            label = key_labels[i]
            database[key]['annotations'] = {'label': label}
        else:
            database[key]['annotations'] = {}

    return database


def load_labels(train_csv_path):
    data = pd.read_csv(train_csv_path)
    return data['label'].unique().tolist()


def convert_kinetics_csv_to_json(train_csv_path, val_csv_path, test_csv_path,
                                 video_dir_path, video_type, dst_json_path):
    labels = load_labels(train_csv_path)
    train_database = convert_csv_to_dict(train_csv_path, 'training')
    val_database = convert_csv_to_dict(val_csv_path, 'validation')
    if test_csv_path.exists():
        test_database = convert_csv_to_dict(test_csv_path, 'testing')

    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)
    if test_csv_path.exists():
        dst_data['database'].update(test_database)

    for k, v in dst_data['database'].items():
        if 'label' in v['annotations']:
            label = v['annotations']['label']
        else:
            label = 'test'

        if video_type == 'jpg':
            video_path = video_dir_path / label / k
            if video_path.exists():
                n_frames = get_n_frames(video_path)
                v['annotations']['segment'] = (1, n_frames + 1)
        else:
            video_path = video_dir_path / label / f'{k}.hdf5'
            if video_path.exists():
                n_frames = get_n_frames_hdf5(video_path)
                v['annotations']['segment'] = (0, n_frames)

    with dst_json_path.open('w') as dst_file:
        json.dump(dst_data, dst_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_path',
                        default='./',
                        type=Path,
                        help=('Directory path including '
                              'kinetics_train.csv, kinetics_val.csv, '
                              '(kinetics_test.csv (optional))'))
    parser.add_argument(
        'n_classes',
        default=2,
        type=int,
        help='400, 600, or 700 (Kinetics-400, Kinetics-600, or Kinetics-700)')
    parser.add_argument('video_path',
                        default='/DATA/disk1/machinelp/3D-ResNets-PyTorch/qpark_action_2/kinetics_videos/jpg',
                        type=Path,
                        help=('Path of video directory (jpg or hdf5).'
                              'Using to get n_frames of each video.'))
    parser.add_argument('video_type',
                        default='jpg',
                        type=str,
                        help=('jpg or hdf5'))
    parser.add_argument('dst_path',
                        default='./kinetics.json',
                        type=Path,
                        help='Path of dst json file.')

    args = parser.parse_args()

    assert args.video_type in ['jpg', 'hdf5']

    train_csv_path = (args.dir_path /
                      'train.csv'.format(args.n_classes))
    val_csv_path = (args.dir_path /
                    'val.csv'.format(args.n_classes))
    test_csv_path = (args.dir_path /
                     'test.csv'.format(args.n_classes))

    convert_kinetics_csv_to_json(train_csv_path, val_csv_path, test_csv_path,
                                 args.video_path, args.video_type,
                                 args.dst_path)


# python kinetics_json_pp.py './' 2 '/DATA/disk1/machinelp/3D-ResNets-PyTorch/qpark_action_2/kinetics_videos/jpg' 'jpg' './kinetics.json'
'''
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --root_path ./qpark_action_2 --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --model resnet \
--model_depth 50 --n_classes 2 --batch_size 32 --n_threads 8 --checkpoint 5 --pretrain_path "r3d50_KM_200ep.pth" > train.log 2>&1 &
'''


