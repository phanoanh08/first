import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PNRec(nn.Module):
    def __init__(self, cfg):
        super(PNRec, self).__init__()

        self.title_encoder = TitleEncoder(cfg)
        self.news_encoder = NodesEncoder(cfg)
        self.cfg = cfg
        self.policy_pos_s = nn.Sequential(
            nn.Linear(cfg.hidden_size * 2, cfg.hidden_size),
            nn.Tanh(),
            nn.Linear(cfg.hidden_size, 1),)
 

        self.news_embedding = nn.Embedding(cfg.tracks_num, cfg.hidden_size)

        self.title_self_attend = SelfAttend(cfg.hidden_size)

    def forward(self, data, test_mode=False):
        neg_num = self.cfg.neg_count
        if test_mode:
            neg_num = 0

        target_news = data[3].reshape(-1, self.cfg.max_lyric_len)
        target_news = self.title_encoder.encode_news(target_news).reshape(-1, neg_num + 1, self.cfg.hidden_size)
        target_all = target_news

        pos_his = data[4].reshape(-1, self.cfg.max_lyric_len)
        pos_his = self.title_encoder.encode_news(pos_his).reshape(-1, self.cfg.pos_hist_length, self.cfg.hidden_size)

        title_v = self.title_self_attend(pos_his)
        title_v = title_v.repeat(1, neg_num + 1).view(-1, neg_num + 1, self.cfg.hidden_size)
        
        pos_s= self.news_encoder(pos_his)

        pos_s = pos_s.repeat(1, neg_num + 1).view(-1, neg_num + 1, self.cfg.hidden_size)
     
        news_states = torch.cat([title_v, target_news], dim=-1)
        gamma_1 = self.policy_pos_s(news_states)
        news_final = gamma_1 * pos_s

        ###return torch.sum(torch.cat([news_final, node_final], dim=-1) * target_all, dim=-1)
        return torch.sum(news_final * target_all, dim=-1)