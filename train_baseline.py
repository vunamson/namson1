import argparse
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, ConcatDataset

import os
import json
import resource
import sys
import pickle

sys.path.insert(1, 'src')
from model import Model
from utils import *
from data import *
from time import time


def create_arg_parser():
    """Create argument parser for our baseline. """
    parser = argparse.ArgumentParser('GMFbaseline')
    
    # DATA  Arguments
    parser.add_argument('--data_dir', help='dataset directory', type=str, default='DATA/')
    parser.add_argument('--tgt_market', help='specify a target market name', type=str, default='t2')
    parser.add_argument('--src_markets', help='specify none ("none") or a few source markets ("-" seperated)to augment the data for training', type=str, default='s1')
    parser.add_argument('--use_qrel', help='merge the valid_qrel into the train dataset, use when final submit', type=bool, default=False)
    parser.add_argument('--tgt_market_valid', help='specify validation run file for target market', type=str, default='DATA/t1/valid_run.tsv')
    parser.add_argument('--tgt_market_test', help='specify test run file for target market', type=str, default='DATA/t1/test_run.tsv') 
    
    parser.add_argument('--exp_name', help='name the experiment',type=str, default='baseline_toy')
    
    parser.add_argument('--train_data_file', help='the file name of the train data',type=str, default='train_5core.tsv') #'train.tsv' for the original data loading
    #parser.add_argument('-- fastmode', help='True if you don\'t want to predict the test set, only valid set', type=bool, default=True)
    
    # MODEL arguments 
    parser.add_argument('--alias', type=str, default='gmf', help='type of model used to train' )
    parser.add_argument('--pretrain', type=str, default=None, help='load pretrain model')
    parser.add_argument('--idbank_pretrain', type=str, default=None, help='load pretrain id_bank')
    parser.add_argument('--freeze_bottom', type=bool, default=False, help='freeze the embedding layers to keep generalization')
    parser.add_argument('--num_epoch', type=int, default=25, help='number of epoches')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0055, help='learning rate')
    parser.add_argument('--l2_reg', type=float, default=1e-06, help='l2 regularization')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--latent_dim', type=int, default=8, help='latent dimensions')
    parser.add_argument('--latent_dim_mlp', type=int, default=8, help='latent dimensions for mlp model')
    parser.add_argument('--num_negative', type=int, default=4, help='num of negative samples during training')
    parser.add_argument('--sample_func', type=object, default = lambda : 0, help='how to sample negative rating' )
    parser.add_argument('--mlp_layers', type=int, nargs='+', default=[16, 64, 32, 16, 8], help='layers config for MLP model')

    parser.add_argument('--cuda', action='store_true', help='use of cuda')
    parser.add_argument('--seed', type=int, default=42, help='manual seed init')
    
    return parser



def build(args):

    set_seed(args)
    
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.set_device(0)
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print(f'Running experiment on device: {args.device}')
    
    ############
    ## Target Market data
    ############
    if args.idbank_pretrain is not None:
        with open(args.idbank_pretrain, 'rb') as centralid_file:
            my_id_bank = pickle.load(centralid_file)
    else:
        my_id_bank = Central_ID_Bank()
    
    train_file_names = args.train_data_file # 'train_5core.tsv', 'train.tsv' for the original data loading
    
    tgt_train_data_dir = os.path.join(args.data_dir, args.tgt_market, train_file_names)
    tgt_train_ratings = pd.read_csv(tgt_train_data_dir, sep='\t')

    print(f'Loading target market {args.tgt_market}: {tgt_train_data_dir}')
    tgt_task_generator = TaskGenerator(my_id_bank, tgt_train_data_dir, use_qrel=args.use_qrel, sample_func=args.sample_func)

    print('Loaded target data!\n')
    valid_qrel_name = os.path.join(args.data_dir, args.tgt_market, 'valid_qrel.tsv')
    tgt_valid_ratings = pd.read_csv(valid_qrel_name, sep='\t')
    tgt_vl_generator = TaskGenerator(my_id_bank, valid_qrel_name, valid = True)
    task_valid_all = {
        0: tgt_vl_generator
    }
    valid_tasksets = MetaMarket_Dataset(task_valid_all, num_negatives=0, meta_split='train' )
    valid_dataloader = MetaMarket_DataLoader(valid_tasksets, sample_batch_size=args.batch_size, shuffle=True, num_workers=0)
    # task_gen_all: contains data for all training markets, index 0 for target market data
    task_gen_all = {
        0: tgt_task_generator
    }  
    
    ############
    ## Source Market(s) Data
    ############
    src_market_list = args.src_markets.split('-')
    if 'none' not in src_market_list:
        cur_task_index = 1
        for cur_src_market in src_market_list:
            cur_src_data_dir = os.path.join(args.data_dir, cur_src_market, train_file_names)
            print(f'Loading {cur_src_market}: {cur_src_data_dir}')
            cur_src_task_generator = TaskGenerator(my_id_bank, cur_src_data_dir,rename=None, use_qrel=True, sample_func=args.sample_func)
            task_gen_all[cur_task_index] = cur_src_task_generator
            cur_task_index+=1
        print('Loaded source data!\n')

    train_tasksets = MetaMarket_Dataset(task_gen_all, num_negatives=args.num_negative, meta_split='train' )
    train_dataloader = MetaMarket_DataLoader(train_tasksets, sample_batch_size=args.batch_size, shuffle=True, num_workers=0)
    tgt_valid_dataloader = tgt_task_generator.instance_a_market_valid_dataloader(args.tgt_market_valid, args.batch_size)
    tgt_test_dataloader = tgt_task_generator.instance_a_market_valid_dataloader(args.tgt_market_test, args.batch_size)
    
    ############
    ## Model  
    ############
    mymodel = Model(args, my_id_bank)
    if args.pretrain is not None:
        mymodel.load(args.pretrain)
        if args.freeze_bottom:
            #mymodel.model.gmf_embedding_user.weight.requires_grad = False
            mymodel.model.gmf_embedding_item.weight.requires_grad = False
            #mymodel.model.mlp_embedding_user.weight.requires_grad = False
            mymodel.model.mlp_embedding_item.weight.requires_grad = False
        else:
            mymodel.model.gmf_embedding_user.weight.requires_grad = True
            mymodel.model.gmf_embedding_item.weight.requires_grad = True
            mymodel.model.mlp_embedding_user.weight.requires_grad = True
            mymodel.model.mlp_embedding_item.weight.requires_grad = True
    #mymodel.fit(train_dataloader, valid_dataloader)
    mymodel.fit(task_gen_all, valid_dataloader)
    ############
    ## Validation and Test Run
    ############
    
    print('Run output files:')
    # validation data prediction
    valid_run_mf = mymodel.predict(tgt_valid_dataloader)
    valid_output_file = os.path.join('baseline_outputs', args.exp_name, args.tgt_market, 'valid_pred.tsv')
    print(f'--validation: {valid_output_file}')
    write_run_file(valid_run_mf, valid_output_file)
#    if args.fastmode == False:
        # test data prediction
    # test_run_mf = mymodel.predict(tgt_test_dataloader)
   #     test_output_file = os.path.join('baseline_outputs', args.exp_name, args.tgt_market, 'test_pred.tsv')
    #    print(f'--test: {test_output_file}')
     #   write_run_file(test_run_mf, test_output_file)
    
    # print evaluation
    print('Evaluating the validation set\n ')
    valid_qrel_mf = read_qrel_file(os.path.join('DATA', args.tgt_market, 'valid_qrel.tsv'))
    task_ov_val, _ = get_evaluations_final(valid_run_mf, valid_qrel_mf)
    for score_name in ['ndcg_cut_10', 'recall_10']:
        print("======= Set val : score(" + score_name + ")=%0.12f =======" % task_ov_val[score_name])
    print('===============\nExperiment finished successfully!')
    return task_ov_val['ndcg_cut_10']
if __name__=="__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    build(args)
