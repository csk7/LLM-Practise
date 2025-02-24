import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# 0. Constants
batch_size = 4
block_size = 8
embedding_dim = 32
max_iters = 1000
learning_rate = 1e-2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 1. Get the data
with open('data/input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

train_data = torch.tensor(encode(text[:int(0.9*len(text))]), device=device)
val_data = torch.tensor(encode(text[int(0.9*len(text)):]), device=device)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    random_indices = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in random_indices])
    y = torch.stack([data[i+1:i+block_size+1] for i in random_indices])
    return x, y

# 2. Create the model
class biGramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx, targets=None):
        #B, T = idx.shape
        logits = self.token_embedding_table(idx) # (B, T) -> (B, T, C)
        logits = self.lm_head(logits) # (B, T, C) -> (B, T, vocab_size)
        if(targets is None):
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

#Model setup
model = biGramLanguageModel(vocab_size)
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

#Training loop
for iter in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(iter % 100 == 0):
        print(f"iter {iter}: loss {loss.item()}")       

