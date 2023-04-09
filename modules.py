import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttend(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super(SelfAttend, self).__init__()

        self.h1 = nn.Sequential(
            nn.Linear(embedding_size, 200),
            nn.Tanh()
        )
        
        self.gate_layer = nn.Linear(200, 1)

    def forward(self, seqs, seq_masks=None):
        """
        :param seqs: shape [batch_size, seq_length, embedding_size]
        :param seq_lens: shape [batch_size, seq_length]
        :return: shape [batch_size, seq_length, embedding_size]
        """
        gates = self.gate_layer(self.h1(seqs)).squeeze(-1)
        if seq_masks is not None:
            gates = gates.masked_fill(seq_masks == 0, -1e9)
        p_attn = F.softmax(gates, dim=-1)
        p_attn = p_attn.unsqueeze(-1)
        h = seqs * p_attn
        output = torch.sum(h, dim=1)
        return output

class TitleEncoder(nn.Module):
    def __init__(self, cfg):
        super(TitleEncoder, self).__init__()
        self.cfg = cfg
        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(cfg.word_emb), freeze=False)

        self.mh_self_attn = nn.MultiheadAttention(
            cfg.hidden_size, num_heads=cfg.head_num
        )
        self.word_self_attend = SelfAttend(cfg.hidden_size)

        self.user_mh_self_attn = nn.MultiheadAttention(
            cfg.hidden_size, num_heads=cfg.head_num
        )
        self.pos_self_attend = SelfAttend(cfg.hidden_size)

        self.dropout = nn.Dropout(cfg.dropout)
        self.word_layer_norm = nn.LayerNorm(cfg.hidden_size)
        self.user_layer_norm = nn.LayerNorm(cfg.hidden_size)

    def _extract_hidden_rep(self, seqs):
        """
        Encoding
        :param seqs: [*, seq_length]
        :param seq_lens: [*]
        :return: Tuple, (1) [*, seq_length, hidden_size] (2) [*, seq_length];
        """
        embs = self.word_embedding(seqs)
        X = self.dropout(embs)

        X = X.permute(1, 0, 2)
        output, _ = self.mh_self_attn(X, X, X)
        output = output.permute(1, 0, 2)
        output = self.dropout(output)
        X = X.permute(1, 0, 2)

        return self.word_layer_norm(output + X)

    def encode_news(self, seqs):
        """
        Args:
            seqs: [*, max_news_len]
            seq_lens: [*]
        Returns:
            [*, hidden_size]
        """
        hiddens = self._extract_hidden_rep(seqs)

        # [*, hidden_size]
        self_attend = self.word_self_attend(hiddens)

        return self_attend

    def encode_user(self, seqs):
        """
        Args:
            seqs: [*, max_hist_len, hidden_size]
        Returns:
            [*, hidden_size]
        """
        user_mh_self_attn = self.user_mh_self_attn
        news_self_attend = self.pos_self_attend

        hiddens = seqs.permute(1, 0, 2)
        user_hiddens, _ = user_mh_self_attn(hiddens, hiddens, hiddens)
        user_hiddens = user_hiddens.permute(1, 0, 2)

        residual_sum = self.user_layer_norm(user_hiddens + seqs)
        user_title_hidden = news_self_attend(residual_sum)

        return user_title_hidden


class NodesEncoder(nn.Module):
    def __init__(self, cfg):
        super(NodesEncoder, self).__init__()
        self.cfg = cfg
        
        self.user_mh_self_attn = nn.MultiheadAttention(
            cfg.hidden_size, num_heads=cfg.head_num)
        
        self.nodes_mh_self_attn = nn.MultiheadAttention(
            cfg.hidden_size, num_heads=cfg.head_num)
        
        self.pos_self_attend = SelfAttend(cfg.hidden_size)

        self.dropout = nn.Dropout(cfg.dropout)
        self.user_layer_norm = nn.LayerNorm(cfg.hidden_size)

    def forward(self, pos): ## neg, pos_nodes, neg_nodes
        """
        Args:
            seqs: [*, max_hist_len, hidden_size]
        Returns:
            [*, hidden_size]
        """

        pos_permuted = pos.permute(1, 0, 2)
        pos_hiddens, _ = self.user_mh_self_attn(pos_permuted, pos_permuted, pos_permuted)
        pos_hiddens = pos_hiddens.permute(1, 0, 2)
        pos_residual = self.user_layer_norm(pos_hiddens + pos)

        pos_s = self.pos_self_attend(pos_residual)


        return pos_s ## pos_s, pos_s_nodes, neg_s, neg_s_nodes, pos_c, pos_c_nodes, neg_c, neg_c_nodes

class MaskedSelfAttend(nn.Module):
    def __init__(self, hidden_size, mask_len) -> None:
        super(MaskedSelfAttend, self).__init__()

        # self.query = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        # self.key = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        # self.value = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.mask = nn.Parameter(torch.eye(mask_len) == 1, requires_grad=False)
        self.hidden_size = hidden_size

    def forward(self, q):
        # q (batch_size, seq_len, hidden_size)
        
        k = q.permute(0, 2, 1)
        sim = torch.matmul(q, k) / math.sqrt(self.hidden_size)
        sim = torch.softmax(sim.masked_fill_(self.mask, -1e9), dim=-1)
        output = torch.matmul(sim, q)

        return output

class Multihead_bandti(nn.Module):

    def __init__(self, cfg):

        super(Multihead_bandti, self).__init__()

        self.head_num = cfg.head_num
        self.head_dim = cfg.hidden_size // cfg.head_num
        self.hidden_size = cfg.hidden_size
        
        self.policy_1 = nn.Sequential(
            nn.Linear(cfg.hidden_size * 2, cfg.hidden_size),
            nn.Tanh(),
            nn.Linear(cfg.hidden_size, self.head_num))

    
    def forward(self, refer, s1, s2, s3, s4):

        gamma_1 = self.policy_1(refer).unsqueeze(-1)

        s1 = s1.view(-1, refer.size(1), self.head_num, self.head_dim)
        final = gamma_1 * s1
        final = final.reshape(-1, refer.size(1), self.hidden_size)

        return final