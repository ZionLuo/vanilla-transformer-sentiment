from embedding_encoder import *
from textprocess_dataloader import *
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import  DataLoader

import matplotlib.pyplot as plt
import pandas as pd

train_file = r'D:\deep_learning\mine\train.csv'
df_train = pd.read_csv(train_file)

target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

target = df_train[target_columns]
df_train = df_train.drop(target_columns, axis=1)

train_croups = df_train['comment_text'].tolist()

config = {
    'num_encoder_layers': 2,  # 新增参数
    'd_model': 16,  # 增大模型尺寸
    'nhead': 4,
    'dim_feedforward': 16,  # 增大前馈网络尺寸
    'dropout': 0.1,
    'batch_size': 16,
    'lr': 1e-2,
    'clip_grad': 1.0,
    'epochs': 2,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

processor = TextProcessor(train_croups)
word_to_idx = processor.build_vocab(top_k=10000)
vocab_size = processor.vocab_size
indices = processor.sentences_to_indices()
labels = target.values
dataset = TextDataset(indices, labels)
dataloader = DataLoader(
    dataset,
    batch_size=config['batch_size'],
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=False
)
print(len(labels))
print(config['device'])

embedding = TransformerEmbeddings(vocab_size, config['d_model']).to(config['device'])

encoder = TransformerEncoder(
    num_layers=config['num_encoder_layers'],  # 新增参数
    d_model=config['d_model'],
    nhead=config['nhead'],
    dim_feedforward=config['dim_feedforward'],
    dropout=config['dropout']
).to(config['device'])

optimizer = optim.Adam(
    list(embedding.parameters()) +
    list(encoder.parameters()),
    lr=config['lr']
)
criterion = nn.BCEWithLogitsLoss()

train_loss = []
for epoch in range(config['epochs']):
    total_loss = 0
    for batch, (inputs, lengths, labels) in enumerate(dataloader):
        inputs = inputs.to(config['device'])
        labels = labels.to(config['device'])

        batch_size, seq_len = inputs.shape

        # 转换为序列优先格式
        src = inputs.transpose(0, 1)  # (seq_len, batch)
        # 前向传播
        optimizer.zero_grad()

        # 编码器
        src_emb = embedding(src)
        output = encoder(src_emb)
        # 计算损失
        loss = criterion(output, labels.float())

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(embedding.parameters()) +
            list(encoder.parameters()),
            config['clip_grad']
        )
        optimizer.step()

        total_loss += loss.item()

        # 每100个batch打印一次
        if batch % 50 == 0:
            print(f'Epoch {epoch + 1} | Batch {batch} | Loss: {loss.item():.4f}')

    print(f'Epoch {epoch + 1} | Avg Loss: {total_loss / len(dataloader):.4f}')
    train_loss.append(loss.item())

torch.save({
    'embedding': embedding.state_dict(),
    'encoder': encoder.state_dict(),
    'processor': processor,
    'word_to_idx': word_to_idx,
    'config': config
}, f'transformer_model.pth')

plt.figure(figsize=(16,9))
plt.plot(train_loss)
plt.show()