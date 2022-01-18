import os
import torch
import random
import resource
import pandas as pd
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset

random.seed(0)


class Central_ID_Bank(object):
    """
    Central for all cross-market user and items original id and their corrosponding index values
    """
    def __init__(self):
        self.user_id_index = {}
        self.item_id_index = {}
        self.last_user_index = 0
        self.last_item_index = 0
        
    def query_user_index(self, user_id):
        if user_id not in self.user_id_index:
            self.user_id_index[user_id] = self.last_user_index
            self.last_user_index += 1
        return self.user_id_index[user_id]
    
    def query_item_index(self, item_id):
        if item_id not in self.item_id_index:
            self.item_id_index[item_id] = self.last_item_index
            self.last_item_index += 1
        return self.item_id_index[item_id]
    
    def query_user_id(self, user_index):
        user_index_id = {v:k for k, v in self.user_id_index.items()}
        if user_index in user_index_id:
            return user_index_id[user_index]
        else:
            print(f'USER index {user_index} is not valid!')
            return 'xxxxx'
        
    def query_item_id(self, item_index):
        item_index_id = {v:k for k, v in self.item_id_index.items()}
        if item_index in item_index_id:
            return item_index_id[item_index]
        else:
            print(f'ITEM index {item_index} is not valid!')
            return 'yyyyy'

    
    

class MetaMarket_DataLoader(object):
    """Data Loader for a few markets, samples task and returns the dataloader for that market"""
    
    def __init__(self, task_list, sample_batch_size, task_batch_size=2, shuffle=True, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
        
        self.num_tasks = len(task_list)
        self.task_list = task_list
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.sample_batch_size = sample_batch_size
        self.task_list_loaders = {
            idx:DataLoader(task_list[idx], batch_size=sample_batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory) \
            for idx in range(len(self.task_list))
        }
        self.task_list_iters = {
            idx:iter(self.task_list_loaders[idx]) \
            for idx in range(len(self.task_list))
        }
        self.task_batch_size = min(task_batch_size, self.num_tasks)
    
    def refresh_dataloaders(self):
        self.task_list_loaders = {
            idx:DataLoader(self.task_list[idx], batch_size=self.sample_batch_size, shuffle=self.shuffle, num_workers=self.num_workers, pin_memory=False) \
            for idx in range(len(self.task_list))
        }
        self.task_list_iters = {
            idx:iter(self.task_list_loaders[idx]) \
            for idx in range(len(self.task_list))
        }
        
    def get_iterator(self, index):
        return self.task_list_iters[index]
        
    def sample_task(self):
        sampled_task_idx = random.randint(0, self.num_tasks-1)
#         print(f'task number {sampled_task_idx} sampled')
        return self.task_list_loaders[sampled_task_idx]
    
    def __len__(self):
        return self.num_tasks
    
    def __getitem__(self, index):
        return self.task_list_loaders[index]
    

        
        
class MetaMarket_Dataset(object):
    """
    Wrapper around market data (task)
    ratings: {
      0: us_market_gen,
      1: de_market_gen,
      ...
    }
    """
    def __init__(self, task_gen_dict, num_negatives=4, meta_split='train'):
        self.num_tasks = len(task_gen_dict)
        if meta_split=='train':
            self.task_gen_dict = {idx:cur_task.instance_a_market_train_task(idx, num_negatives) for idx, cur_task  in task_gen_dict.items()}
        else:
            self.task_gen_dict = {idx:cur_task.instance_a_market_valid_task(idx, split=meta_split) for idx, cur_task  in task_gen_dict.items()}
        
    def __len__(self):
        return self.num_tasks

    def __getitem__(self, index):
        return self.task_gen_dict[index]
        


class MarketTask(Dataset):
    """
    Individual Market data that is going to be wrapped into a metadataset  i.e. MetaMarketDataset

    Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset
    """
    def __init__(self, task_index, user_tensor, item_tensor, target_tensor):
        """
        args:

            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.task_index = task_index
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

        
    def __len__(self):
        return self.user_tensor.size(0)
    
    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]


    

class TaskGenerator(object):
    """Construct dataset"""

    def __init__(self, id_index_bank, fname=None, sample_func=lambda: 0, rename=None, use_qrel=False, valid=False):
        """
        args:
            train_data: pd.DataFrame, which contains 3 columns = ['userId', 'itemId', 'rating']
            id_index_bank: converts ids to indices 
        """

        self.id_index_bank = id_index_bank
        self.valid = valid
        self.sample_func = sample_func
        # None for evaluation purposes
        if fname is not None: 
            self.dir = fname.split('/')[:-1]
            self.ratings = pd.read_csv(fname, sep='\t')
            if fname.split('/')[-1] == 'train.tsv':
                self.ratings['rating'] = self.ratings['rating']/5.0
            if use_qrel:
                qrel_ratings = pd.read_csv(os.path.join(*self.dir, 'valid_qrel.tsv'), sep='\t')
                qrel_ratings['rating'] = qrel_ratings['rating'].astype(float)
                self.ratings = pd.concat([self.ratings, qrel_ratings])
                self.ratings.drop_duplicates(inplace=True)
            if rename is not None:
                self.ratings.userId = self.ratings['userId'].apply(lambda x: rename + x[2:])

            # get item and user pools
            self.user_pool_ids = set(self.ratings['userId'].unique())
            self.item_pool_ids = set(self.ratings['itemId'].unique())

            # replace ids with corrosponding index for both users and items
            self.ratings['userId'] = self.ratings['userId'].apply(lambda x: self.id_index_bank.query_user_index(x) )
            self.ratings['itemId'] = self.ratings['itemId'].apply(lambda x: self.id_index_bank.query_item_index(x) )

            # get item and user pools (indexed version)
            self.user_pool = set(self.ratings['userId'].unique())
            self.item_pool = set(self.ratings['itemId'].unique())

            # create negative item samples
            self.negatives_train = self._sample_negative0() ## Sử dụng hàm sample negative
            self.train_ratings = self.ratings
        
    
    def _sample_negative(self): # sample ngẫu nhiên negative trong tập 99valid_run  ---> better (just in valid)
        neg_samples = open(os.path.join(*self.dir, 'valid_run.tsv'))
        pos_samples = pd.read_csv(os.path.join(*self.dir, 'valid_qrel.tsv'), sep='\t')
        # bỏ các sample positive ra khỏi các simple negative từ tập 100valid_run
        negatives_train = {}
        for line in neg_samples:
            linetoks = line.split('\t')
            neg_user_id = self.id_index_bank.query_user_index(linetoks[0])
            neg_item_ids = list(map(self.id_index_bank.query_item_index, linetoks[1].strip().split(',')))
            negatives_train[int(neg_user_id)] = neg_item_ids
        pos_samples['userId'] = pos_samples['userId'].apply(lambda x: self.id_index_bank.query_user_index(x) )
        pos_samples['itemId'] = pos_samples['itemId'].apply(lambda x: self.id_index_bank.query_item_index(x) )
        for row in pos_samples.itertuples(): 
            negatives_train[int(row.userId)].remove(row.itemId)
        neg_samples.close()
        return negatives_train

    def _sample_negative2(self): # sample ngẫu nhiên negative từ tập train
        neg_test = open(os.path.join(*self.dir, 'valid_run.tsv'))
        pos_valid = pd.read_csv(os.path.join(*self.dir, 'valid_qrel.tsv'), sep='\t')#need rename
        negatives_train = {}
        for line in neg_test:
            linetoks = line.split('\t')
            user_id = self.id_index_bank.query_user_index(linetoks[0])
            neg_item_ids = set(map(self.id_index_bank.query_item_index, linetoks[1].strip().split(',')))
            negatives_train[user_id] = neg_item_ids

        by_userid_group = self.ratings.groupby("userId")['itemId']
        for userid, group_frame in by_userid_group:
            pos_itemids = set(group_frame.values.tolist())
            neg_itemids = self.item_pool - pos_itemids - negatives_train[userid]
            neg_itemids_train = neg_itemids
            negatives_train[userid] = neg_itemids_train
        return negatives_train
        
    def _sample_negative0(self): # sample từ toàn bộ dữ liệu
        by_userid_group = self.ratings.groupby("userId")['itemId']
        negatives_train = {}
        for userid, group_frame in by_userid_group:
            pos_itemids = set(group_frame.values.tolist())
            neg_itemids = self.item_pool - pos_itemids
            neg_itemids_train = neg_itemids
            negatives_train[userid] = neg_itemids_train
        return negatives_train

    def instance_a_market_train_task(self, index, num_negatives):
        """instance train task's torch Dataset"""
        users, items, ratings = [], [], []
        train_ratings = self.train_ratings
        for row in train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            #if (row.rating > 3.7):
            #    ratings.append(1.0)
            #else:
            #    ratings.append(0.1)
            #ratings.append(float(row.rating))
            ratings.append(0.95)
            
            cur_negs = self.negatives_train[int(row.userId)]
            cur_negs = random.sample(cur_negs, min(num_negatives, len(cur_negs)) )
            for neg in cur_negs:
                users.append(int(row.userId))
                items.append(int(neg))
                #ratings.append(float(0))  # negative samples get 0 rating
                if self.valid:
                    ratings.append(0.5)
                else:
                    ratings.append(self.sample_func()) # get triangular random random.triangular(0,0.4,0.1)

        dataset = MarketTask(index, user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        return dataset
    
    
    def instance_a_market_train_dataloader(self, index, num_negatives, sample_batch_size, shuffle=True, num_workers=0):
        """instance train task's torch Dataloader"""
        dataset = self.instance_a_market_train_task(index, num_negatives)
        return DataLoader(dataset, batch_size=sample_batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)
        
    
    def load_market_valid_run(self, valid_run_file):
        users, items, ratings = [], [], []
        with open(valid_run_file, 'r') as f:
            for line in f:
                linetoks = line.split('\t')
                user_id = linetoks[0]
                item_ids = linetoks[1].strip().split(',')
                for cindex, item_id in enumerate(item_ids):
                    users.append(self.id_index_bank.query_user_index(user_id))
                    items.append(self.id_index_bank.query_item_index(item_id))
                    ratings.append(float(0))

        dataset = MarketTask(0, user_tensor=torch.LongTensor(users),
                                            item_tensor=torch.LongTensor(items),
                                            target_tensor=torch.FloatTensor(ratings))
        return dataset
    
    def instance_a_market_valid_dataloader(self, valid_run_file, sample_batch_size, shuffle=False, num_workers=0):
        """instance target market's validation data torch Dataloader"""
        dataset = self.load_market_valid_run(valid_run_file)
        return DataLoader(dataset, batch_size=sample_batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)

    
