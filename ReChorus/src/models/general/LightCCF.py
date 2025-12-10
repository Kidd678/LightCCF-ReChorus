# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.BaseModel import GeneralModel

class LightCCF(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'ssl_lambda', 'tau', 'reg_lambda']

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
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        GeneralModel.__init__(self, args, corpus)
        self.emb_size = args.emb_size
        self.ssl_lambda = args.ssl_lambda
        self.tau = args.tau
        self.reg_lambda = args.reg_lambda

        # Define embeddings (MF style, no GCN)
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        
        # Initialize weights (Xavier Uniform is standard for these models)
        nn.init.xavier_uniform_(self.u_embeddings.weight)
        nn.init.xavier_uniform_(self.i_embeddings.weight)

    def forward(self, feed_dict):
        self.check_list = []
        u_ids = feed_dict['user_id'] # [batch_size]
        i_ids = feed_dict['item_id'] # [batch_size, 1 + num_neg]

        u_emb = self.u_embeddings(u_ids)           # [batch_size, emb_size]
        i_emb = self.i_embeddings(i_ids)           # [batch_size, 1 + num_neg, emb_size]

        # Prediction: Dot product
        # u_emb: [B, E] -> [B, 1, E]
        # i_emb: [B, N, E]
        # result: [B, N]
        prediction = (u_emb[:, None, :] * i_emb).sum(dim=-1)

        return {
            'prediction': prediction,
            'u_emb': u_emb,
            'i_emb': i_emb
        }

    def loss(self, out_dict):
        # 1. BPR Loss
        # GeneralModel.loss uses 'prediction' to calculate BPR loss automatically
        bpr_loss = super().loss(out_dict)

        u_emb = out_dict['u_emb']
        i_emb = out_dict['i_emb'] # [B, 1+neg, E]

        # 2. Regularization Loss
        # Penalize all embeddings used in the batch
        # u_emb: [B, E]
        # i_emb: [B, N, E] -> flatten to [B*N, E] for norm calculation
        
        reg_loss = (1/2) * u_emb.norm(2).pow(2) / float(u_emb.shape[0]) + \
                   (1/2) * i_emb.norm(2).pow(2) / float(i_emb.shape[0] * i_emb.shape[1])
        
        reg_loss = reg_loss * self.reg_lambda

        # 3. Neighborhood Aggregation (NA) Loss
        # Only use Positive Items for NA loss usually
        u_emb_norm = F.normalize(u_emb, p=2, dim=1)
        i_pos_emb = i_emb[:, 0, :] # [B, E]
        i_pos_norm = F.normalize(i_pos_emb, p=2, dim=1)

        na_loss = self.get_neighbor_aggregate_loss(u_emb_norm, i_pos_norm, self.tau) * self.ssl_lambda

        return bpr_loss + reg_loss + na_loss

    def get_neighbor_aggregate_loss(self, embedding1, embedding2, tau):
        # embedding1: User embeddings [B, E] (Normalized)
        # embedding2: Positive Item embeddings [B, E] (Normalized)
        
        # Positive score: exp(sim(u, i_pos) / tau)
        # sim is dot product of normalized embeddings (cosine similarity)
        pos_score = (embedding1 * embedding2).sum(dim=-1)
        pos_score = torch.exp(pos_score / tau)

        # Denominator: 
        # sim(u, all_i_pos_in_batch) + sim(u, all_u_in_batch)
        # This implementation follows the LightCCF paper/code where they use
        # in-batch negatives from both users and items.
        
        # matmul(e1, e2.T) -> [B, B] matrix where [i, j] is sim(u_i, i_pos_j)
        total_score_items = torch.matmul(embedding1, embedding2.transpose(0, 1))
        
        # matmul(e1, e1.T) -> [B, B] matrix where [i, j] is sim(u_i, u_j)
        total_score_users = torch.matmul(embedding1, embedding1.transpose(0, 1))
        
        # Remove self-loops for users? 
        # In contrastive learning, usually we mask out the self-similarity for the denominator if included.
        # But LightCCF code implementation:
        # total_score = torch.exp(total_score_items / tau).sum(dim=1) + torch.exp(total_score_users / tau).sum(dim=1)
        # It seems to include everything.
        
        # However, for stability, let's stick to the structure that worked in the GCN version 
        # which mimicked the official code logic.
        
        total_score = torch.exp(total_score_items / tau).sum(dim=1) + \
                      torch.exp(total_score_users / tau).sum(dim=1)
        
        # Loss: -log(pos_score / total_score)
        na_loss = -torch.log(pos_score / total_score + 1e-6)
        return torch.mean(na_loss)