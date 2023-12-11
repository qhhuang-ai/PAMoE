import os
import argparse
from datetime import datetime
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn


# path to a pretrained word embedding file
word_emb_path = '/home/henry/glove/glove.840B.300d.txt'
assert (word_emb_path is not None)

username = Path.home().name
project_dir = Path(__file__).resolve().parent.parent
sdk_dir = project_dir.joinpath('CMU-MultimodalSDK')
data_dir = str(project_dir) + "/src/data/"
# data_dir = '/home/lzh/codes/'
# /home/lzh/codes/MOSEI/
data_dict = {'mosi': data_dir +'MOSI', 'mosei': data_dir +
    'MOSEI', 'ur_funny': data_dir + 'UR_FUNNY'}
optimizer_dict = {'RMSprop': optim.RMSprop, 'Adam': optim.Adam}
activation_dict = {'elu': nn.ELU, "hardshrink": nn.Hardshrink, "hardtanh": nn.Hardtanh,
                   "leakyrelu": nn.LeakyReLU, "prelu": nn.PReLU, "relu": nn.ReLU, "rrelu": nn.RReLU,
                   "tanh": nn.Tanh}

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
}

criterion_dict = {
    'mosi': 'L1Loss',
    'iemocap': 'CrossEntropyLoss',
    'ur_funny': 'CrossEntropyLoss'
}


def get_args():
    parser = argparse.ArgumentParser(description='MOSI-and-MOSEI Sentiment Analysis')
    parser.add_argument('-f', default='', type=str)

    # Tasks
    parser.add_argument('--dataset', type=str, default='mosei', choices=['mosi', 'mosei'],
                        help='dataset to use (default: mosi)')
    # parser.add_argument('--data_path', type=str, default='/home/lzh/datasets/MSA/MOSEI/',
    #                     help='path for storing the dataset')
    parser.add_argument('--tv', type=float, default=0.1,help= 'loss')
    parser.add_argument('--ta', type=float, default=0.1, help='loss')
    parser.add_argument('--va_in', type=float, default=0.1, help='loss')
    parser.add_argument('--la_in', type=float, default=0.1, help='loss')
    parser.add_argument('--neg', type=float, default=1, help='loss')
    parser.add_argument('--neu', type=float, default=1, help='loss')
    parser.add_argument('--heat', type=float, default=0.8, help='loss')
    parser.add_argument('--aux', type=float, default=0, help='loss')
    parser.add_argument('--noise_aux', type=float, default=0, help='loss')
    parser.add_argument('--polarity', type=float, default=0.1, help='loss')

    parser.add_argument('--choose', type=str, default="self", help='output operation')
    parser.add_argument('--interaction', type=str, default="add", help='output operation')
    parser.add_argument('--fin_interaction', type=str, default="add", help='output operation')
    parser.add_argument('--mode', type=str, default="last", help='output operation')
    parser.add_argument('--out', type=str, default="cat", help='output operation')

    parser.add_argument('--expert_polarity', type=str, default="h", help='label for expert')
    parser.add_argument('--expert', type=str, default="CONSMLP", help='expert type')
    parser.add_argument('--expert_num', type=int, default=3, help='expert num')
    parser.add_argument('--k', type=int, default=2, help='activated expert num')

    parser.add_argument('--condrop', type=float, default=0.1, help='dropout for expert')
    parser.add_argument('--conln', type=int, default=1, help='LN for expert')

    parser.add_argument('--layer_l', type=int, default=1, help='CMA layer for t')
    parser.add_argument('--attn_dropout', type=float, default=0, help='dropout for CMA-t layer')
    parser.add_argument('--attn_dropout_a', type=float, default=0, help='dropout for CMA-a layer')
    parser.add_argument('--layer_a', type=int, default=1, help='CMA layer for a')
    parser.add_argument('--attn_dropout_v', type=float, default=0, help='dropout for CMA-v layer')
    parser.add_argument('--layer_v', type=int, default=1, help='CMA layer for v')

    parser.add_argument('--pretrain', type=int, default=1, help='mode for label')
    parser.add_argument('--load', type=int, default=1, help='load for bert')
    parser.add_argument('--fixed', type=int, default=1, help='fix label')


    # Dropouts
    parser.add_argument('--dropout_a', type=float, default=0.1,
                        help='dropout of acoustic LSTM out layer')
    parser.add_argument('--dropout_v', type=float, default=0.1,
                        help='dropout of visual LSTM out layer')
    parser.add_argument('--dropout_prj', type=float, default=0.1,
                        help='dropout of projection layer')

    parser.add_argument('--relu_dropout', type=float, default=0.1,
                        help='relu dropout')
    parser.add_argument('--res_dropout', type=float, default=0.1,
                        help='residual block dropout')
    parser.add_argument('--out_dropout', type=float, default=0.0,
                        help='output layer dropout')
    parser.add_argument('--embed_dropout', type=float, default=0.25,
                        help='embedding dropout')
    parser.add_argument('--attn_mask', action='store_false',
                        help='use attention mask for Transformer (default: true)')

    parser.add_argument('--vonly', action='store_true',
                        help='use the crossmodal fusion into v (default: False)')
    parser.add_argument('--aonly', action='store_true',
                        help='use the crossmodal fusion into a (default: False)')
    parser.add_argument('--lonly', action='store_true',
                        help='use the crossmodal fusion into l (default: False)')

    parser.add_argument('--layers', type=int, default=5,
                        help='number of layers in the network (default: 5)')
    parser.add_argument('--num_heads', type=int, default=5,
                        help='number of heads for the transformer network (default: 5)')

    # Architecture
    parser.add_argument('--n_tv', type=int, default=0,
                        help='number of V-T transformer  (default: 0)')
    parser.add_argument('--n_ta', type=int, default=1,
                        help='number of A-T transformer (default: 1)')

    parser.add_argument('--multiseed', action='store_true', help='training using multiple seed')
    parser.add_argument('--contrast', action='store_false', help='using contrast')
    parser.add_argument('--add_va', action='store_false', help='va or not')
    parser.add_argument('--n_layer', type=int, default=1,
                        help='number of layers ')
    parser.add_argument('--cpc_layers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--d_vh', type=int, default=16,
                        help='hidden size in visual rnn')
    parser.add_argument('--d_ah', type=int, default=16,
                        help='hidden size in acoustic rnn')
    parser.add_argument('--d_vout', type=int, default=16,
                        help='output size in visual rnn')
    parser.add_argument('--d_aout', type=int, default=16,
                        help='output size in acoustic rnn')
    parser.add_argument('--bidirectional', action='store_true', help='Whether to use bidirectional rnn')
    parser.add_argument('--d_prjh', type=int, default=128,
                        help='hidden size in projection network')
    parser.add_argument('--pretrain_emb', type=int, default=768,
                        help='dimension of pretrained model output')

    # Activations
    parser.add_argument('--mmilb_mid_activation', type=str, default='ReLU',
                        help='Activation layer')
    parser.add_argument('--mmilb_last_activation', type=str, default='Tanh',
                        help='Activation layer')
    parser.add_argument('--cpc_activation', type=str, default='Tanh',
                        help='Activation layer')

    # Training Setting
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size (default: 32)')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='gradient clip value (default: 0.8)')
    parser.add_argument('--lr_main', type=float, default=1e-3,
                        help='initial learning rate(default: 1e-3)')
    parser.add_argument('--lr_bert', type=float, default=5e-5,
                        help='initial learning rate (default: 5e-5)')
    parser.add_argument('--lr_mmilb', type=float, default=1e-3,
                        help='initial learning rate (default: 1e-3)')
    parser.add_argument('--alpha', type=float, default=0.1, help='weight')
    parser.add_argument('--beta', type=float, default=0.1, help='weight')
    parser.add_argument('--mem', type=float, default=1, help='memorys')
    parser.add_argument('--weight_decay_main', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')  # ！！！！ 越大训练越不好  正则  小了
    parser.add_argument('--weight_decay_bert', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')
    parser.add_argument('--weight_decay_club', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')

    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='number of epochs (default: 40)')
    parser.add_argument('--when', type=int, default=20,
                        help='when decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='when stop ')
    parser.add_argument('--update_batch', type=int, default=1,
                        help='update batch ')

    # Logistics
    parser.add_argument('--log_interval', type=int, default=100,
                        help='frequency of result logging (default: 100)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    args = parser.parse_args()

    valid_partial_mode = args.lonly + args.vonly + args.aonly

    if valid_partial_mode == 0:
        args.lonly = args.vonly = args.aonly = True
    elif valid_partial_mode != 1:
        raise ValueError("You can only choose one of {l/v/a}only.")

    return args


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, data, mode='train'):
        """Configuration Class: set kwargs as class attributes with setattr"""
        self.dataset_dir = data_dict[data.lower()]
        self.sdk_dir = sdk_dir
        self.mode = mode
        # Glove path
        self.word_emb_path = word_emb_path

        # Data Split ex) 'train', 'valid', 'test'
        self.data_dir = self.dataset_dir

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        config_str = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(dataset='mosi', mode='train', batch_size=32):
    config = Config(data=dataset, mode=mode)

    config.dataset = dataset
    config.batch_size = batch_size

    return config
