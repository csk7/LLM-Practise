import torch
import torch.nn
from model.model import Transformer

from train.train import train

if __name__ == '__main__':
    src_vocab_size = 50
    tgt_vocab_size = 50

    d_model = 16

    num_heads = 2

    num_layers = 3

    d_ff = 32

    max_seq_length = 10

    dropout = 0.1

    if(torch.cuda.is_available()):
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    transformer = Transformer(src_vocab_size=src_vocab_size,tgt_vocab_size=tgt_vocab_size, d_model=d_model, num_heads=num_heads,
                              num_layers=num_layers, d_ff=d_ff, max_seq_length=max_seq_length, dropout = dropout)
    
    transformer = transformer.to(device)
    
    

    train(transformer, src_vocab_size, tgt_vocab_size, d_model, d_ff, num_heads, num_layers, max_seq_length, dropout, device)