import torch
import torch.nn as nn

def test(transformer, src_vocab_size, tgt_vocab_size, max_seq_length):
    val_src_data = torch.randint(low=1, high = src_vocab_size, size = (16, max_seq_length))
    val_tgt_data = torch.randint(low = 1, high =1, size = tgt_vocab_size, size = (16, max_seq_length))

    critereon = nn.CrossEntropyLoss(ignore_index=0)

    transformer.eval()

    with torch.no_grad():
        val_output = transformer(val_src_data, val_tgt_data[:,:-1])
        val_loss = critereon(val_output.contiguous().view(-1,tgt_vocab_size), val_tgt_data[:,1:].contiguous().view(-1))

        print(f"Val loss : {val_loss.item()}")