import torch
import torch.nn as nn

import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        #Ensure model dimension d is visible by number of heads
        assert d_model % num_heads == 0, 'd_model must be divisible by number of heads'

        #Initialize dimensions
        #Embedding size = d
        #position embedding size = d

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads


        self.W_q = nn.Linear(in_features = d_model, out_features= d_model)
        self.W_k = nn.Linear(in_features = d_model, out_features = d_model)
        self.W_v = nn.Linear(in_features = d_model, out_features=d_model)

        self.W_o = nn.Linear(in_features = d_model, out_features = d_model)

    def scaled_dot_product(self, K, Q, V, mask =None):
        #Calculate attention_scores
        attn_scores = torch.matmul(Q, K.transpose(-2,-1))/math.sqrt(self.d_k)
        ##PRINT Here the dimesnions

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, value= -1e9)

        attn_prob = torch.softmax(attn_scores, dim = -1)

        output = torch.matmul(attn_prob, V)

        return output
    
    def split_heads(self, x):
        #Reshape the input to have num-heads for multi head attaention
        batch_size, seq_lenth, d_model = x.size()
        return x.view(batch_size, seq_lenth, self.num_heads, self.d_k).transpose(1,2)
    
    def combine_heads(self, x):
        #combine multiple heads bck to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, seq_length,  self.d_model)
    
    def forward(self, Q, K, V, mask = None):
        #Apply linear transformations and split_heads

        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product(K, Q, V, mask)

        output = self.W_o(self.combine_heads(attn_output))
        return output
    

class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForwardNetwork,self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*-(math.log(10000) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position*div_term)

        self.register_buffer('pe', pe.unsqueeze(0)) #Register buffer that should not be considered as a model parameter

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer,self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForwardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)

        self.feed_forward = PositionWiseFeedForwardNetwork(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x,x,x,tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))

        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model,
                 num_heads, d_ff, num_layers, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encode_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decode_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        #Need to see what exactly this function does
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2) #src masks padding 0
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)

        seq_length = tgt.size(1)
        nopeak_mask = (1-torch.triu(torch.ones(1,seq_length,seq_length), diagonal=1)).bool().to('cuda:0')
        tgt_mask = tgt_mask & nopeak_mask

        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encode_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decode_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
    
    