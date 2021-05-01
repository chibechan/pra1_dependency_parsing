from transformers import AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from model.BERT import BERT_Parsing
from utils.preprocess_dep import DepCollate, DepDataset
from utils.preprocess_KWDLC import make_df_KWDLC

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device", DEVICE)

# load df
df = make_df_KWDLC("./data/KWDLC-1.0/dat/rel/")
train_df = df.loc[:int(len(df) * 0.8), :]
valid_df = df.loc[int(len(df)*0.8):int(len(df)*0.9), :]

# define parameter
'''
max_len: BERTの入力最大長
batch_size: batch_size
max_epochs: 最大epoch数
num_training_steps: trainingのstep数
num_warmup_steps: warmupのstep数
learning_rate: 学習率
dep_dim: dependencyのモデルにおける隠れそうのサイズ
savePATH: modelをsaveする先
BPE: BPEありかなしか
valid_step: このstep数ごとにvalidationを行う
'''
max_len = 320
batch_size = 12
max_epochs = 2
num_training_steps = max_epochs * int(len(train_df)/batch_size)
num_warmup_steps = int(num_training_steps*0.1)
learning_rate = 2e-5
dep_dim = 256
BPE = True
valid_step = 1000

# path
if BPE is True:
    model_path = 'data/NICT_BERT-base_JapaneseWikipedia_32K_BPE/'
    savePATH = 'drive/My Drive/google_colab/save_model/BPE_model'
else:
    model_path = 'data/NICT_BERT-base_JapaneseWikipedia_100K/'
    savePATH = 'drive/My Drive/google_colab/save_model/Without_BPE_model'
vocab_path = model_path+'vocab.txt'

# loaders
train_dataset = DepDataset(train_df, vocab_path,
                          max_len=max_len, BPE=BPE)
valid_dataset = DepDataset(valid_df, vocab_path,
                          max_len=max_len, BPE=BPE)
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          collate_fn=DepCollate, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                          collate_fn=DepCollate)

# download model
model = BERT_Parsing(model_path=model_path, dep_dim=dep_dim)
model = model.to(DEVICE)


# define optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer
                if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps)


# training
print("Start Training!")
n_epochs = 0
step = 0
train_loss = 0
model.train()

while True:
    for inputs, masks, segments, keiDep_matrixs, mapping_matrixs, kmasks, \
      paths, keiDep in tqdm(train_loader, desc='Training', leave=False):

        optimizer.zero_grad()
        loss, prediction, _ = \
            model(inputs, token_type_ids=segments, attention_mask=masks,
                  mapping_matrix=mapping_matrixs, s_mask=kmasks,
                  dep=keiDep_matrixs, dep_list=keiDep, ud=False)

        step += 1
        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # validation
        if (step % valid_step == 0):
            model.eval()
            correct_num = 0
            num = 0
            valid_loss = total = 0
            with torch.no_grad():
                for inputs, masks, segments, keiDep_matrixs, \
                   mapping_matrixs, kmasks, paths, keiDep \
                   in tqdm(valid_loader, desc='Validating', leave=False):

                    loss, prediction = model(
                        inputs, token_type_ids=segments, attention_mask=masks,
                        mapping_matrix=mapping_matrixs, s_mask=kmasks,
                        dep=keiDep_matrixs, dep_list=keiDep, ud=False)[:2]

                    valid_loss += loss.item()
                    total += 1
                    correct_num += int(torch.sum(prediction*keiDep_matrixs))
                    num += int(torch.sum(keiDep_matrixs))
            train_loss = train_loss/valid_step
            valid_loss = valid_loss/total
            acc = correct_num/num
            print(f'epoch #{n_epochs + 1:3d}\ttrain_loss: {train_loss:.3f}\t \
                                              valid_loss: {valid_loss:.3f}\n',)
            print('validation accuracy is ', acc)

            # trainの再スタート
            model.train()
            train_loss = 0

    n_epochs += 1
    if n_epochs >= max_epochs:
        break

# save_model
torch.save(model, savePATH)
