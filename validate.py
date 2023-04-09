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
import torch.nn.functional as F
import logging
from torch.utils.data import Dataset, TensorDataset, DataLoader
import math


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def run(dev_dataset, epoch):
    filenum=20
    batch_size=128
    root="data"
    start_dev=2
    result_path = 'result/test/'
    saved_model_path = 'checkpoint/model.ep{}'.format(epoch)

    model_cfg = ModelConfig(root)
    model = PNRec(model_cfg)

    pretrained_model = torch.load(saved_model_path)

    model.load_state_dict(pretrained_model['model_state_dict'], strict=False)
    model.eval()
    
    valid_data_loader = DataLoader(dev_dataset, shuffle=False)

    data_iter = tqdm(enumerate(valid_data_loader),
                        desc="epoch_dev %d" % epoch,
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

        with open(result_path + 'tmp_test-{}.json'.format(epoch), 'w+', encoding='utf-8') as f:
            json.dump(tmp_dict, f)

    gather(result_path, 'tmp_test-{}.json'.format(epoch), epoch, validate=True, save=True)


if __name__ == "__main__":
    test = torch.load('data/test/test.pt')
    for i in range(5):
        save_model_path = 'checkpoint/model.ep{}'.format(i)
        run(test, i)

        