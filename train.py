import torch
import logging
import argparse
import numpy as np
import pandas as pd
import openpyxl
import config
import os
from tqdm import tqdm
from qadataset import QADataSet
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from robertamodel import RoBERTaModel

os.environ["CUDA_VISIBLE_DEVICES"]= "1"

#update dataset
train_df = pd.read_excel(config.TRAIN_DATA)
df = train_df.astype({'Question': 'str','Answer': 'str'})

parser = argparse.ArgumentParser(description=config.TRAIN_DESCRIPTION)

parser.add_argument("--batch_size", 
                    type=int,
                    default=config.TRAIN_BATCH_SIZE, 
                    help=config.TRAIN_BATCH_HELP)
parser.add_argument("--n_epoch", 
                    type=int,
                    default=config.TRAIN_EPOCH, 
                    help=config.TRAIN_EPOCH_HELP)
parser.add_argument("--lr",
                    type=float, 
                    default=config.TRAIN_LR,
                    help=config.TRAIN_LR_HELP)

args = parser.parse_args()

batch_size = args.batch_size
n_epoch = args.n_epoch
lr = args.lr

model = RoBERTaModel()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
tokenizer = AutoTokenizer.from_pretrained(config.ROBERTA_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_dataset = QADataSet(df)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

losses = []
epoch_ = []
model.train()
for epoch in range(n_epoch):
    count = 0
    epoch_loss = 0.0
    for questions, answers in tqdm(train_loader):
        optimizer.zero_grad()
        questions = torch.stack(questions)
        answers = torch.stack(answers)
        input_ids = questions.transpose(1,0)
        label = answers.transpose(1,0)
        input_ids = input_ids.to(device)
        label = label.to(device)
        loss = model(input_ids, label).loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        count += 1
        
    epoch_loss = epoch_loss / count

    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

    epoch_.append(epoch)
    losses.append(epoch_loss)
    
    print('Epoch {0}/{1} Average Loss: {2}'.format(epoch+1, n_epoch, epoch_loss))