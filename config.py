import json
import pickle
import numpy as np

class ModelConfig():
    def __init__(self, root):

        tracks_dict = json.load(open('{}/tracks_dict.jsonl'.format(root), 'r', encoding='utf-8'))
        self.tracks_num = len(tracks_dict)
        self.word_emb = np.load('{}/emb.npy'.format(root))
        self.word_num = len(self.word_emb)

        self.pos_hist_length = 30
        self.max_lyric_len = 100
        self.neg_count = 4
        self.word_dim = 300
        self.hidden_size = 300
        self.head_num = 6
        self.dropout = 0.2
        
        return None