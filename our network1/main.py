from __future__ import absolute_import
import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from utilsgai import SG
from worksgai import train
import warnings

warnings.filterwarnings('ignore')


def main(params, model):
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.exists('records'):
        os.mkdir('records')
    dir = os.path.dirname(os.path.abspath(__file__)).replace('\\','/') + '/'       # 返回seg的绝对路径

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs',      type=int,   default=300,    help='Number of epochs to train')
    parser.add_argument('--validation_step', type=int,   default=1,      help='How often to perform validation (epochs)')
    parser.add_argument('--crop_height',     type=int,   default=224,    help='Height of resized input image to network')
    parser.add_argument('--crop_width',      type=int,   default=224,    help='Width of resized input image to network')
    parser.add_argument('--batch_size',      type=int,   default=1,      help='Number of images in each batch')
    parser.add_argument('--learning_rate',   type=float, default=0.001,  help='learning rate used for train')
    parser.add_argument('--num_workers',     type=int,   default=4,      help='num of workers')
    parser.add_argument('--num_classes',     type=int,   default=3,      help='num of object classes (with void)')
    parser.add_argument('--use_gpu',         type=bool,  default=True,   help='whether to user gpu for training')
    parser.add_argument('--model_name',      type=str,   default=None,   help='path to model')
    parser.add_argument('--save_each_model', type=bool,  default=False,  help='whether to save all models')
    parser.add_argument('--test_pic_name',   type=str,   default=None,   help='which picture to test')
    parser.add_argument('--base_dir',        type=str,   default=dir,    help='project directory')
    parser.add_argument('--data_name',       type=str,   default=None,   help='data directory')
    args = parser.parse_args(params)

    parser_ = argparse.ArgumentParser()
    parser_.add_argument('-cuda',            type=int,   default=0,      help='choose GPU ID')
    args_ = parser_.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args_.cuda)

    assert args.data_name is not None, 'Please input dataset.'
    assert args.model_name is not None, 'Please name model.'
    assert args.test_pic_name is not None, 'Please input test picture.'

    train_path = args.base_dir + f'datasets/{args.data_name}/train/image/'           # 训练集图片存放地址
    train_label_path = args.base_dir + f'datasets/{args.data_name}/train/label/'     # 训练集图片存放地址
    val_path = args.base_dir + f'datasets/{args.data_name}/val/image/'               # 训练集图片存放地址
    val_label_path = args.base_dir + f'datasets/{args.data_name}/val/label/'         # 验证集标签存放地址
    csv_path = args.base_dir + f'datasets/{args.data_name}/class_dict.csv'           # 标签分类的颜色值存放地址

    dataset_train = SG(train_path, train_label_path, csv_path)
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    dataset_val = SG(val_path, val_label_path, csv_path)
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=args.num_workers)

    miou_path = os.listdir(args.base_dir + 'checkpoints')
    if args.data_name not in miou_path:
        os.mkdir(args.base_dir + f'checkpoints/{args.data_name}')
    miou_path_list = os.listdir(args.base_dir + f'checkpoints/{args.data_name}')
    if args.model_name not in miou_path_list:
        os.mkdir(args.base_dir + f'checkpoints/{args.data_name}/{args.model_name}')

    demo_path = os.listdir(args.base_dir + 'demo')
    if args.data_name not in demo_path:
        os.mkdir(args.base_dir + f'demo/{args.data_name}')
    demo_path_list = os.listdir(args.base_dir + f'demo/{args.data_name}')
    if args.model_name not in demo_path_list:
        os.mkdir(args.base_dir + f'demo/{args.data_name}/{args.model_name}')

    records_path = os.listdir(args.base_dir + 'records')
    if args.data_name not in records_path:
        os.mkdir(args.base_dir + f'records/{args.data_name}')

    datetime = time.strftime("%Y%m%d%H%M", time.localtime())
    train(args, model, dataloader_train, dataloader_val, csv_path, datetime)


if __name__ == '__main__':
    print('==========Main==========')

    from models.enet import ENet
    from models.espnetv2 import ESPNetv2
    from models.dfn import DFN
    from models.q import Net
    from models.hrnet import HRNet
    from models.pan import PAN
    from models.ACFnet import Seg_Model
    from models.dabnet import DABNet
    from models.cvtnet import CvT
    from models.ghostnet import GhostNet
    from models.unet import UNet
    from models.my_42_5simamcbam import Pspcbamcat42_5



    # model 1
    params_1 = [
        '--num_epochs', '300',
        '--crop_height', '256',
        '--crop_width', '256',
        '--learning_rate', '0.001',
        '--num_workers', '2',
        '--num_classes', '2',
        '--batch_size', '12',
        '--save_each_model', False,
        '--test_pic_name', 'test_L',             # demo原图名称
        '--data_name', 'L',                      # 数据集名称
        '--model_name', 'my_42_5simamcbamLfh'                     # 模型保存名称
    ]
    model_1 = Net()
    main(params_1, model_1)






