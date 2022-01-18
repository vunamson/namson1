import torch
import pickle
from utils import *
from time import time
from data import *

class Model(object):
    def __init__(self, args, my_id_bank):
        self.args = args
        self.my_id_bank = my_id_bank
        self.model = self.prepare_model()
        
    
    def prepare_model(self):
        if self.my_id_bank is None:
            print('ERR: Please load an id_bank before model preparation!')
            return None
        model_alias = self.args.alias #gmf
        self.config = {'batch_size': self.args.batch_size, #1024,
              'optimizer': self.args.optimizer,
              'tgt_market': self.args.tgt_market, #'t1'
              'adam_lr': self.args.lr, #0.005, #1e-3,
              'sgd_lr':self.args.lr,
              'sgd_momentum': 0.9,
              'latent_dim': self.args.latent_dim, #8
              'latent_dim_mlp': self.args.latent_dim_mlp, #8
              'num_negative': self.args.num_negative, #4
              'l2_regularization': self.args.l2_reg, #1e-07,
              'use_cuda': torch.cuda.is_available() and self.args.cuda, #False,
              'device_id': 0,
              'embedding_user': None,
              'embedding_item': None,
              'save_trained': True,
              'num_users': int(self.my_id_bank.last_user_index+1), 
              'num_items': int(self.my_id_bank.last_item_index+1),
              'mlp_layers': self.args.mlp_layers #[16 64 32 16 8]
        }
        if model_alias == 'gmf':
            self.model = GMF(self.config)
        elif model_alias == 'nmf':
            self.model = NMF(self.config)
        elif model_alias  == 'mlp':
            self.model = MLP(self.config)
        print(f'Model is {model_alias.upper()}!')
        self.model = self.model.to(self.args.device)
        print(self.model)
        return self.model
    
    def fit(self, task_gen_all, valid_dataloader):
    #def fit(self, train_dataloader, valid_dataloader): 
        opt = use_optimizer(self.model, self.config)
        loss_func = torch.nn.BCELoss()
        ############
        ## Train
        ############
        #self.model.train()
        #valid_qrel_name = os.path.join(self.args.data_dir, self.config['tgt_market'], 'valid_qrel.tsv')
        #tgt_valid_ratings = pd.read_csv(valid_qrel_name, sep='\t')
         
        for epoch in range(self.args.num_epoch):
            self.model.train()
            tr_time = time()
            total_loss = 0
            print('Epoch {} starts !'.format(epoch))
            
            # train the model for some certain iterations
            #train_dataloader.refresh_dataloaders()
            train_tasksets = MetaMarket_Dataset(task_gen_all, num_negatives=self.args.num_negative, meta_split='train' )
            train_dataloader = MetaMarket_DataLoader(train_tasksets, sample_batch_size=self.args.batch_size, shuffle=True, num_workers=0)
            data_lens = [len(train_dataloader[idx]) for idx in range(train_dataloader.num_tasks)]
            iteration_num = max(data_lens)
            nums_batch = 0
            for iteration in range(iteration_num):
                for subtask_num in range(train_dataloader.num_tasks): # get one batch from each dataloader
                    cur_train_dataloader = train_dataloader.get_iterator(subtask_num)
                    try:
                        train_user_ids, train_item_ids, train_targets = next(cur_train_dataloader)
                    except:
                        new_train_iterator = iter(train_dataloader[subtask_num])
                        train_user_ids, train_item_ids, train_targets = next(new_train_iterator)
                    
                    train_user_ids = train_user_ids.to(self.args.device)
                    train_item_ids = train_item_ids.to(self.args.device)
                    train_targets = train_targets.to(self.args.device)
                
                    opt.zero_grad()
                    ratings_pred = self.model(train_user_ids, train_item_ids)
                    loss = loss_func(ratings_pred.view(-1), train_targets)
                    loss.backward()
                    opt.step()    
                    total_loss += loss.item()
                    nums_batch += 1
            print('Total Train Loss: ', total_loss/nums_batch, ' Time: ', time()-tr_time)
            self.calc_valid_loss(valid_dataloader, loss_func)        
            
            #sys.stdout.flush()
            print('-' * 80)
        
        print('Model is trained! and saved at:')
        self.save()

    def calc_valid_loss(self, valid_dataloader, loss_func):
        vl_time = time()
        vl_loss = 0
        nums_batch = 0
        self.model.eval()
        valid_dataloader.refresh_dataloaders()
        data_lens = [len(valid_dataloader[idx]) for idx in range(valid_dataloader.num_tasks)]
        iteration_num = max(data_lens)
        for iteration in range(iteration_num):
            for subtask_num in range(valid_dataloader.num_tasks): # get one batch from each dataloader
                cur_valid_dataloader = valid_dataloader.get_iterator(subtask_num)
                try:
                    valid_user_ids, valid_item_ids, valid_targets = next(cur_valid_dataloader)
                except:
                    new_valid_iterator = iter(valid_dataloader[subtask_num])
                    valid_user_ids, valid_item_ids, valid_targets = next(new_valid_iterator)

                valid_user_ids = valid_user_ids.to(self.args.device)
                valid_item_ids = valid_item_ids.to(self.args.device)
                valid_targets = valid_targets.to(self.args.device)

                with torch.no_grad():
                    ratings_pred = self.model(valid_user_ids, valid_item_ids)
                    loss = loss_func(ratings_pred.view(-1), valid_targets)
                    vl_loss += loss.item() 
                    nums_batch += 1
            
        print('Total Valid Loss: ', vl_loss/nums_batch, ' Time: ', time()-vl_time)    
        
        
    # produce the ranking of items for users
    def predict(self, eval_dataloader):
        stime = time()
        self.model.eval()
        task_rec_all = []
        task_unq_users = set()
        for test_batch in eval_dataloader:
            test_user_ids, test_item_ids, test_targets = test_batch
    
            cur_users = [user.item() for user in test_user_ids]
            cur_items = [item.item() for item in test_item_ids]
            
            test_user_ids = test_user_ids.to(self.args.device)
            test_item_ids = test_item_ids.to(self.args.device)
            test_targets = test_targets.to(self.args.device)

            with torch.no_grad():
                batch_scores = self.model(test_user_ids, test_item_ids)
                batch_scores = batch_scores.detach().cpu().numpy()

            for index in range(len(test_user_ids)):
                task_rec_all.append((cur_users[index], cur_items[index], batch_scores[index][0].item()))

            task_unq_users = task_unq_users.union(set(cur_users))

        task_run_mf = get_run_mf(task_rec_all, task_unq_users, self.my_id_bank)
        print('Predict time: ', time() - stime)
        return task_run_mf
    
    ## SAVE the model and (idbank,args config)
    def save(self):
        if self.config['save_trained']:
            model_dir = f'checkpoints/{self.args.tgt_market}_{self.args.src_markets}_{self.args.exp_name}.model'
            cid_filename = f'checkpoints/{self.args.tgt_market}_{self.args.src_markets}_{self.args.exp_name}.pickle'
            print(f'--model: {model_dir}')
            print(f'--id_bank: {cid_filename}')
            torch.save(self.model.state_dict(), model_dir)
            with open(cid_filename, 'wb') as centralid_file:
                pickle.dump(self.my_id_bank, centralid_file)
    
    ## LOAD the model and idbank
    def load(self, checkpoint_dir):
        model_dir = checkpoint_dir
        state_dict = torch.load(model_dir, map_location=self.args.device)
        self.model.load_state_dict(state_dict, strict=False)
        print(f'Pretrained weights from {model_dir} are loaded!')
        




class GMF(torch.nn.Module):
    def __init__(self, config):
        super(GMF, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.trainable_user = False
        self.trainable_item = False

        if config['embedding_user'] is None:
            self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
            self.trainable_user = True
        else:
            self.embedding_user = config['embedding_user']
            
        if config['embedding_item'] is None:
            self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
            self.trainable_item = True
        else:
            self.embedding_item = config['embedding_item']
        
        #self.user_biases = torch.nn.Embedding(self.num_users, 1)
        #self.item_biases = torch.nn.Embedding(self.num_items, 1)

        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        if self.trainable_user:
            user_embedding = self.embedding_user(user_indices)
        else:
            user_embedding = self.embedding_user[user_indices]
        if self.trainable_item:
            item_embedding = self.embedding_item(item_indices)
        else:
            item_embedding = self.embedding_item[item_indices]
        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.affine_output(element_product)
        #logits += self.user_biases(user_indices) + self.item_biases(item_indices) 

        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

class NMF(torch.nn.Module):
    def __init__(self, config):
        super(NMF, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.latent_dim_mlp = config['latent_dim_mlp']
        self.mlp_layers = config['mlp_layers']
       
        self.gmf_embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.gmf_embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        self.mlp_embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.mlp_embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        #self.mlp_layer1 = torch.nn.Linear(in_features=self.latent_dim*4, out_features=self.latent_dim*2)
        #self.mlp_layer2 = torch.nn.Linear(in_features=self.latent_dim*2, out_features=self.latent_dim)
        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(self.mlp_layers[:-1], self.mlp_layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        self.affine_output = torch.nn.Linear(in_features=self.latent_dim + self.mlp_layers[-1], out_features=1)
        #self.user_biases = torch.nn.Embedding(self.num_users, 1)
        #self.item_biases = torch.nn.Embedding(self.num_items, 1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        gmf_user_embedding = self.gmf_embedding_user(user_indices)
        gmf_item_embedding = self.gmf_embedding_item(item_indices)
        gmf_vector = torch.mul(gmf_user_embedding, gmf_item_embedding)

        mlp_user_embedding = self.mlp_embedding_user(user_indices)
        mlp_item_embedding = self.mlp_embedding_item(item_indices)
        mlp_vector = torch.concat([mlp_user_embedding, mlp_item_embedding], dim=1)
        #mlp_vector = self.mlp_layer1(mlp_vector)
        #mlp_vector = torch.nn.functional.relu(mlp_vector)
        #mlp_vector = self.mlp_layer2(mlp_vector)
        #mlp_vector = torch.nn.functional.relu(mlp_vector)
        for idx in range(len(self.fc_layers)):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.GELU()(mlp_vector)
            #mlp_vector = torch.nn.BatchNorm1d(self.mlp_layers[idx+1])(mlp_vector)
            mlp_vector = torch.nn.Dropout(p=0.4)(mlp_vector)

        predict_vector = torch.concat([gmf_vector, mlp_vector], dim=1)
        logits = self.affine_output(predict_vector)
        #logits += self.user_biases(user_indices) + self.item_biases(item_indices) ##add bias
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass

class MLP(torch.nn.Module):

    def __init__(self, config):
        """
        Function to initialize the MLP class
        :param config: configuration choice
        """
        super(MLP, self).__init__()

        self.config = config
        # Specify number of users, number of items, and number of latent dimensions
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim_mlp']

        # Generate user embedding
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        # Generate item embedding
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        # Generate a list of Fully-Connected layers
        self.fc_layers = torch.nn.ModuleList()
        # Apply linear transformations between each fully-connected layer
        for idx, (in_size, out_size) in enumerate(zip(config['mlp_layers'][:-1], config['mlp_layers'][1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        # Apply a linear transformation to the incoming last fully-connected layer -> output of size 1
        self.affine_output = torch.nn.Linear(in_features=config['mlp_layers'][-1], out_features=1)
        # Perform sigmoid activation function
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        """
        Function to perform a forward pass for rating prediction
        :param user_indices: a list of user indices
        :param item_indices: a list of item indices
        :return: predicted rating
        """

        # Generate user embedding from user indices
        user_embedding = self.embedding_user(user_indices)
        # Generate item embedding from item indices
        item_embedding = self.embedding_item(item_indices)
        # Concatenate the user and iem embeddings -> Resulting a latent vector
        vector = torch.cat([user_embedding, item_embedding], dim=-1)

        # Go through all fully-connected layers
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            # Perform ReLU activation
            vector = torch.nn.ELU()(vector)
            # vector = torch.nn.BatchNorm1d()(vector)
            # vector = torch.nn.Dropout(p=0.5)(vector)

        # Apply linear transformation to the final vector
        logits = self.affine_output(vector)
        # Apply sigmoid (logistic) to get the final predicted rating
        rating = self.logistic(logits)

        return rating

    def init_weight(self):
        pass