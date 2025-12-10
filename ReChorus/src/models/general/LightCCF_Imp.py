# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.BaseModel import GeneralModel

class LightCCF_Imp(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'ssl_lambda', 'tau', 'reg_lambda', 'noise_eps', 'learnable_tau']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--ssl_lambda', type=float, default=0.1,
                            help='Weight for Neighborhood Aggregation (NA) loss.')
        parser.add_argument('--tau', type=float, default=0.2,
                            help='Temperature for NA loss.')
        parser.add_argument('--reg_lambda', type=float, default=1e-4,
                            help='Weight for L2 regularization.')
        parser.add_argument('--noise_eps', type=float, default=0.0,
                            help='Epsilon for embedding noise perturbation (Innovation).')
        parser.add_argument('--learnable_tau', type=int, default=0,
                            help='Whether to make temperature learnable (Innovation). 1=True, 0=False.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        GeneralModel.__init__(self, args, corpus)
        self.emb_size = args.emb_size
        self.ssl_lambda = args.ssl_lambda
        self.reg_lambda = args.reg_lambda
        self.noise_eps = args.noise_eps
        
        # Learnable Temperature
        if args.learnable_tau:
            self.tau = nn.Parameter(torch.tensor(args.tau))
        else:
            self.tau = args.tau

        # Define embeddings (MF style, no GCN)
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.u_embeddings.weight)
        nn.init.xavier_uniform_(self.i_embeddings.weight)

    def forward(self, feed_dict):
        self.check_list = []
        u_ids = feed_dict['user_id'] # [batch_size]
        i_ids = feed_dict['item_id'] # [batch_size, 1 + num_neg]

        u_emb = self.u_embeddings(u_ids)           # [batch_size, emb_size]
        i_emb = self.i_embeddings(i_ids)           # [batch_size, 1 + num_neg, emb_size]

        prediction = (u_emb[:, None, :] * i_emb).sum(dim=-1)

        return {
            'prediction': prediction,
            'u_emb': u_emb,
            'i_emb': i_emb
        }

    def loss(self, out_dict):
        # 1. BPR Loss
        bpr_loss = super().loss(out_dict)

        u_emb = out_dict['u_emb']
        i_emb = out_dict['i_emb'] # [B, 1+neg, E]

        # 2. Regularization Loss
        reg_loss = (1/2) * u_emb.norm(2).pow(2) / float(u_emb.shape[0]) + \
                   (1/2) * i_emb.norm(2).pow(2) / float(i_emb.shape[0] * i_emb.shape[1])
        
        reg_loss = reg_loss * self.reg_lambda

        # 3. Neighborhood Aggregation (NA) Loss with Noise Perturbation (Innovation)
        # Apply noise to embeddings BEFORE normalization
        
        # Noise generation: uniform random noise scaled by eps
        # We use torch.rand_like which gives [0, 1), center it to [-0.5, 0.5] then scale
        # or just standard normal noise. SimGCL uses uniform * eps.
        
        # User noise
        u_noise = torch.rand_like(u_emb) * self.noise_eps
        u_emb_aug = u_emb + u_noise
        
        # Item noise (only for positive items used in NA loss)
        i_pos_emb = i_emb[:, 0, :] # [B, E]
        i_noise = torch.rand_like(i_pos_emb) * self.noise_eps
        i_pos_emb_aug = i_pos_emb + i_noise

        # Normalize augmented embeddings
        u_emb_norm = F.normalize(u_emb_aug, p=2, dim=1)
        i_pos_norm = F.normalize(i_pos_emb_aug, p=2, dim=1)

        na_loss = self.get_neighbor_aggregate_loss(u_emb_norm, i_pos_norm, self.tau) * self.ssl_lambda

        return bpr_loss + reg_loss + na_loss

    def get_neighbor_aggregate_loss(self, embedding1, embedding2, tau):
        pos_score = (embedding1 * embedding2).sum(dim=-1)
        pos_score = torch.exp(pos_score / tau)
        
        total_score_items = torch.matmul(embedding1, embedding2.transpose(0, 1))
        total_score_users = torch.matmul(embedding1, embedding1.transpose(0, 1))
        
        total_score = torch.exp(total_score_items / tau).sum(dim=1) + \
                      torch.exp(total_score_users / tau).sum(dim=1)
        
        na_loss = -torch.log(pos_score / total_score + 1e-6)
        return torch.mean(na_loss)
