from config import ModelConfig
from pnrec import PNRec
from train_util import save_checkpoint_by_epoch
from gather import gather

import os
import argparse
import json
import pickle
from tqdm import tqdm
import time
import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import logging
from torch.utils.data import Dataset, TensorDataset, DataLoader 
import math
# from random import sample

def run(train_dataset, valid_dataset, is_break=False, model_path=None):
    """
    train and evaluate
    """
    batch_size = 128
    epochs = 10
    lr = 0.001
    weight_decay = 1e-6
    port = 9337
    root = "data"
    mc = ModelConfig(root)
    result_path = 'result/train/'
    checkpoint_path = 'checkpoint/'
    
    # Build Dataloader
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Build model.
    model = PNRec(mc)

    # Build optimizer.
    steps_one_epoch = len(train_data_loader)
    train_steps = epochs * steps_one_epoch
    print("Total train steps: ", train_steps)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    
    # Load model if break
    if is_break:
      checkpoint = torch.load(model_path)
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      epoch_break = checkpoint['epoch']
      print('Load model done!')

    # Training and validation
    for epoch in range(epoch_break + 1, epochs):

        train(epoch, model, train_data_loader, optimizer, steps_one_epoch)
        # save_checkpoint_by_epoch(model.state_dict(), epoch, 'checkpoint/model-{}.pt'.format(epoch)) 
        save_checkpoint_by_epoch(model, optimizer, epoch, checkpoint_path)  
        validate(result_path, epoch, model, valid_data_loader)       
        gather(result_path, 'tmp-{}.json'.format(epoch), epoch, validate=True, save=True)

def train(epoch, model, loader, optimizer, steps_one_epoch):
    """
    train loop
    """
    model.train()
    model.zero_grad()

    for i, data in tqdm(enumerate(loader), total=len(loader), desc="epoch-{} train".format(epoch)):
        if i >= steps_one_epoch:
            break
        
        pred = model(data).squeeze()
        loss = F.cross_entropy(pred, data[1])
        loss.backward()
        optimizer.step()
        model.zero_grad()


def validate(result_path, epoch, model, valid_data_loader, fast_dev=False, top_k=20):

    model.eval()
    # Setting the tqdm progress bar
    data_iter = tqdm(enumerate(valid_data_loader),
                    desc="epoch_test %d" % epoch,
                    total=len(valid_data_loader),
                    bar_format="{l_bar}{r_bar}")
                        
    with torch.no_grad():
        preds, truths, imp_ids = list(), list(), list()
        for i, data in data_iter:

            imp_ids += data[0].numpy().tolist()
            truths += data[1].numpy().tolist()

            pred = model(data, test_mode=True)
            if pred.dim() > 1:
                pred = pred.squeeze()
            try:
                preds += pred.numpy().tolist()
            except:
                preds.append(int(pred.numpy()))

        tmp_dict = {}
        tmp_dict['imp'] = imp_ids
        tmp_dict['labels'] = truths
        tmp_dict['preds'] = preds

        with open(result_path + 'tmp-{}.json'.format(epoch), 'w', encoding='utf-8') as f:
            json.dump(tmp_dict, f)


if __name__ == "__main__":
    trainset = torch.load('data/train/train.pt')
    devset = torch.load('data/dev/dev.pt')
    run(trainset, devset)
    # save_model_path = 'checkpoint/model.ep{}'.format(3)
    # run(trainset, devset, True, save_model_path)
    gather('result/train/', 'tmp-{}.json'.format(0), 0, validate=True, save=True)