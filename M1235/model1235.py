

##############
# 1Ligand Atom
#2 Angle
#3 Protein
#5 Pocket

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


#SMILESCLen = 64
PP_LEN = 40
AL_LEN = 41
LL_LEN=17
PK_LEN = 40
hidden_dim = 512#384
out_channle = 512#384 

class Squeeze(nn.Module):  # Dimention Module
    @staticmethod
    def forward(input_data: torch.Tensor):
        return input_data.squeeze()


class Model1235(nn.Module):
    def __init__(self):
        super().__init__()
        embed_size = 128

        seq_oc = 128
        pkt_oc = 128
        smi_oc = 128

        # SMILES, POCKET, PROTEIN Embedding
        
        
        #self.ll_embed1 = nn.Embedding(SMILESCLen+1, embed_size)
        self.ll_embed2 = nn.Linear(LL_LEN, embed_size)
        self.al_embed = nn.Embedding(AL_LEN+1, embed_size) #self.pl_embed = nn.Embedding(PL_LEN+1, embed_size)
        self.pp_embed = nn.Linear(PP_LEN, embed_size)
        self.pk_embed = nn.Linear(PK_LEN, embed_size)

        
        """self.conv_ll1 = nn.Sequential(
            nn.Conv1d(embed_size, 32, 4),
            nn.PReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 6),
            nn.PReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 8),
            nn.PReLU(),

            nn.AdaptiveMaxPool1d(1),
            Squeeze()
        )"""
        
        self.conv_ll2 = nn.Sequential(
            nn.Conv1d(embed_size, 32, 4),
            nn.PReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 6),
            nn.PReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 8),
            nn.PReLU(),

            nn.AdaptiveMaxPool1d(1),
            Squeeze()
        )

        self.conv_pp = nn.Sequential(
            nn.Conv1d(embed_size, 32, 4),
            nn.PReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 8),
            nn.PReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 12),
            nn.PReLU(),

            nn.AdaptiveMaxPool1d(1),
            Squeeze()
        )
        

        self.conv_pk = nn.Sequential(
            nn.Conv1d(embed_size, 32, 4),
            nn.PReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 6),
            nn.PReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 8),
            nn.PReLU(),

            nn.AdaptiveMaxPool1d(1),
            Squeeze()
        )
        
        self.conv_al = nn.Sequential(
            nn.Conv1d(embed_size, 32, 4),
            nn.PReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 8),
            nn.PReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 12),
            nn.PReLU(),

            nn.AdaptiveMaxPool1d(1),
            Squeeze()
        )


        
        # Dropout
        self.cat_dropout = nn.Dropout(0.2)#0.3 0.2
        # FNN
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 1)
        )
        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim, out_channle))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v) #

        return output

    def attention_net(self, x):
        q = torch.tanh(torch.matmul(x, self.w_omega))
        k = torch.tanh(torch.matmul(x, self.w_omega))
        v = torch.tanh(torch.matmul(x, self.w_omega))

        # Compute the attention output
        context = self.scaled_dot_product_attention(q, k, v)

        # Apply residual connection
        output = x + context

        return output

    def forward(self, data): #batch of data
        
        ll2,al,pk,pp= data 
        #print(" pp, al, ll",  pp.shape, al.shape, ll.shape) # pp, al, ll torch.Size([16, 1000, 58]) torch.Size([16, 1000]) torch.Size([16, 150])
        # TODO:  PP Layers
        pp = pp.to(torch.float32)
        pp_embed = self.pp_embed(pp)
        pp_embed = torch.transpose(pp_embed, 1, 2)
        pp_conv = self.conv_pp(pp_embed)
        
        pk = pp.to(torch.float32)
        pk_embed = self.pk_embed(pp)
        pk_embed = torch.transpose(pk_embed, 1, 2)
        pk_conv = self.conv_pk(pk_embed)


        # TODO: PL layer
        al = al.to(torch.int32)#  al torch.Size([16, 1000])
       
        al_embed = self.al_embed(al)
        #print(" al_embed1",  al_embed.shape)
        al_embed = torch.transpose(al_embed, 1, 2)
        al_conv = self.conv_al(al_embed)
        #print("al_embed", al_embed.shape)"""
        
        
       

        # TODO: LL Layer
        """ll1 = ll1.to(torch.int32)
        ll_embed1 = self.ll_embed1(ll1)
        ll_embed1 = torch.transpose(ll_embed1, 1, 2)
        ll_conv1 = self.conv_ll1(ll_embed1)"""
        
        #print("al1", al.shape)
        
        # TODO: LL Layer
        ll2 = ll2.to(torch.float32)
        ll_embed2 = self.ll_embed2(ll2)
        ll_embed2 = torch.transpose(ll_embed2, 1, 2)
        ll_conv2 = self.conv_ll2(ll_embed2)

        pp_conv = torch.reshape(pp_conv, (-1, 128))
        pk_conv = torch.reshape(pk_conv, (-1, 128))
        al_conv = torch.reshape(al_conv, (-1, 128))
        #ll_conv1 = torch.reshape(ll_conv1, (-1, 128))
        ll_conv2 = torch.reshape(ll_conv2, (-1, 128))
        #print("al2", al.shape)

        # print(pp_conv.shape)
        # print(pl_conv.shape)
        # print(ll_conv.shape)

        concat = torch.cat([pp_conv, al_conv, pk_conv, ll_conv2], dim=1)
        concat = torch.reshape(concat, (concat.shape[0], -1, 128*4))
        
        concat = self.attention_net(concat)
        concat = torch.reshape(concat, (-1, 128*4))
        concat = self.cat_dropout(concat)

        output = self.classifier(concat)
        return output





















