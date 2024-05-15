
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

PP_LEN = 40
SMILESCLen = 64
PK_LEN = 40
AL_LEN = 41
LL_LEN=17
hidden_dim = 640#384
out_channle = 640 #384 

class Squeeze(nn.Module):  # Dimention Module
    @staticmethod
    def forward(input_data: torch.Tensor):
        return input_data.squeeze()


class Model12345(nn.Module):
    def __init__(self):
        super().__init__()
        embed_size = 128

        seq_oc = 128
        pkt_oc = 128
        smi_oc = 128

        # SMILES, POCKET, PROTEIN Embedding
        self.ll_embed1 = nn.Embedding(SMILESCLen+1, embed_size)
        self.ll_embed2 = nn.Linear(LL_LEN, embed_size)
        
        self.al_embed = nn.Embedding(AL_LEN+1, embed_size) #self.pl_embed = nn.Embedding(PL_LEN+1, embed_size)
        self.pk_embed = nn.Linear(PK_LEN, embed_size)
        self.pp_embed = nn.Linear(PP_LEN, embed_size)
        
        self.smi_attention_poc = EncoderLayer(128, 128, 0.1, 0.1, 2) 
        
        #self.smi_attention_poc2 = EncoderLayer(128, 128, 0.1, 0.1, 2) 
        
        
        
        
        


        self.conv_ll1 = nn.Sequential(
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

        
        self.conv_pp = nn.Sequential(
            nn.Conv1d(embed_size, 32, 4),
            nn.PReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 8),
            nn.PReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, 12),
            nn.PReLU(),
            
           # nn.Conv1d(128, 256, 12),
           # nn.PReLU(),

            nn.AdaptiveMaxPool1d(1),
            Squeeze()
        )
        

        # Dropout
        self.cat_dropout = nn.Dropout(0.2)#0.3 0.2
        # FNN
        self.classifier = nn.Sequential(
            #nn.Linear( pkt_oc + smi_oc, 128),
            nn.Linear( 128*5, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, 1)
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
        #pk, ll = data 
        ll1,ll2,al,pk,pp= data 
        
        ll1 = ll1.to(torch.int32)   # ligand SMI
        ll_embed1 = self.ll_embed1(ll1)
        
        ll2 = ll2.to(torch.float32) #ligand atom
        ll_embed2 = self.ll_embed2(ll2)
        
        al = al.to(torch.int32)#  al torch.Size([16, 1000])
        al_embed = self.al_embed(al)
        
        pk = pk.to(torch.float32)
        pk_embed = self.pk_embed(pk)
        
        pp = pp.to(torch.float32)
        pp_embed = self.pp_embed(pp)
        
        #**************************
        ll_attention1=ll_embed1
        ll_embed1 = self.smi_attention_poc(ll_embed1, pk_embed)
        pk_embed = self.smi_attention_poc(pk_embed, ll_attention1)
    
        #***********************
        
        ll_attention2=ll_embed2
        ll_embed2 = self.smi_attention_poc(ll_embed2, al_embed)
        al_embed = self.smi_attention_poc(al_embed, ll_attention2)
        
        #al_embed = self.smi_attention_poc(al_embed, ll_attention2)
        
        
        #print(" al_embed1",  al_embed.shape)
        #print("al_embed", al_embed.shape)
    
        
        ll_embed1 = torch.transpose(ll_embed1, 1, 2)
        ll_conv1 = self.conv_ll1(ll_embed1)
        
        ll_embed2 = torch.transpose(ll_embed2, 1, 2)
        ll_conv2 = self.conv_ll2(ll_embed2)
        #print("al1", al.shape)
        
        al_embed = torch.transpose(al_embed, 1, 2)
        al_conv = self.conv_al(al_embed)
        
        pk_embed = torch.transpose(pk_embed, 1, 2)
        pk_conv = self.conv_pk(pk_embed)
        
        pp_embed = torch.transpose(pp_embed, 1, 2)
        pp_conv = self.conv_pp(pp_embed)

        
        ll_conv1 = torch.reshape(ll_conv1, (-1, 128))
        ll_conv2 = torch.reshape(ll_conv2, (-1, 128))
        al_conv = torch.reshape(al_conv, (-1, 128))
        pk_conv = torch.reshape(pk_conv, (-1, 128))
        pp_conv = torch.reshape(pp_conv, (-1, 128))
        
        #print("al2", al.shape)

        # print(pp_conv.shape)
        # print(pl_conv.shape)
        # print(ll_conv.shape)
        
        concat = torch.cat([ ll_conv1, ll_conv2, al_conv, pk_conv, pp_conv], dim=1)

        #concat = torch.cat([ pk_conv, ll_conv], dim=1)
        concat = torch.reshape(concat, (concat.shape[0], -1, 128*5))
        
        concat = self.attention_net(concat)
        concat = torch.reshape(concat, (-1, 128*5))
        concat = self.cat_dropout(concat)

        output = self.classifier(concat)
        return output

 
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, kv, attn_bias=None):
        y = self.self_attention_norm(x)
        kv = self.self_attention_norm(kv)
        y = self.self_attention(y, kv, kv, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        #        self.gelu = GELU()
        self.gelu = nn.ReLU(inplace=True)
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias
        x = torch.softmax(x, dim=3)
        #
        # Attention analyse
        #        csvwriter = csv.writer(open("attention.csv","a+",newline=""))

        #temp = x.cpu().numpy()
        temp=x
        #print(temp)
        #        temp = temp.argmax(axis = 2)
        temp = temp.mean(axis=2)
#        print(temp.shape)
        if temp.shape == (290,2,63):
            np.save("attention.npy",temp)
         
     

        #        
#        np.save("")
        #        csvwriter.writerows(temp.tolist())
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]
        
        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)
        
        x = self.output_layer(x)
        
        assert x.size() == orig_q_size
        return x



















