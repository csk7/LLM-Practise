import torch
import torch.nn as nn
import torch.optim as optim

def train(transformer, src_vocab_size, tgt_vocab_size, d_model, d_ff, num_heads, num_layers, max_seq_length, dropout, device):
    src_data = torch.randint(low = 1, high = src_vocab_size, size = (1, max_seq_length)) #(batch_size, seq_length)
    tgt_data = torch.randint(low = 1, high = tgt_vocab_size, size = (1, max_seq_length)) #(batch_size, seq_length)

    src_data = src_data.to(device)
    tgt_data = tgt_data.to(device)

    criteron = nn.CrossEntropyLoss(ignore_index=0) ##Will not consider index of 0 as its padding elements
    optimizer = optim.Adam(transformer.parameters(), lr = 0.0001, betas = (0.9, 0.98), eps=1e-9)

    transformer.train()

    for epoch in range(1):
        running_loss = 0.0
        
        optimizer.zero_grad()

        outputs = transformer(src_data, tgt_data[:,:-1]) #Except the last token, common where target is shifted by one token

        loss = criteron(outputs.contiguous().view(-1,tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        print(f"Epoch{epoch+1}, loss = {loss.item()}")






