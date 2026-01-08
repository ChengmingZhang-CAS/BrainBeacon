import torch
from torch import nn
import torch.nn.functional as F
from ..utils.pe import select_pe_encoder
from ..utils import create_norm


class OmicsEmbedder(nn.Module):
    def __init__(self, pretrained_gene_list, bb_gene_emb, num_hid=1024, fix_embedding=False):
        super().__init__()
        self.pretrained_gene_list = pretrained_gene_list
        self.gene_index = dict(zip(pretrained_gene_list, list(range(len(pretrained_gene_list)))))
        if bb_gene_emb is not None:
            self.bb_gene_emb = nn.Parameter(bb_gene_emb, requires_grad=not fix_embedding)
        else:
            self.bb_gene_emb = nn.Parameter(torch.randn([len(pretrained_gene_list), num_hid], dtype=torch.float32) * 0.005)

    def forward(self, x_dict, input_gene_list=None):
        if 'masked_x_seq' in x_dict:
            x = x_dict['masked_x_seq']
        else:
            x = x_dict['x_seq']

        if 'dropout' in x_dict:
            indices = x._indices().t()
            values = x._values()
            temp = values.sum()
            values = values.float()
            values = torch.distributions.binomial.Binomial(values, x_dict['dropout']).sample()
            x = torch.sparse.FloatTensor(indices.t(), values, x.shape)

        x = torch.log1p(x)
        if input_gene_list is not None:
            gene_idx = torch.tensor([self.gene_index[o] for o in input_gene_list if o in self.gene_index]).long()
            x_dict['input_gene_mask'] = gene_idx
        else:
            if x.shape[1] != len(self.pretrained_gene_list):
                raise ValueError(
                    'The input gene size is not the same as the pretrained gene list. '
                    'Please provide the input gene list.'
                )
            gene_idx = torch.arange(x.shape[1]).long()
        gene_idx = gene_idx.to(x.device)
        feat = F.embedding(gene_idx, self.bb_gene_emb)
        feat = torch.sparse.mm(x, feat)
        return feat


class OmicsEmbeddingLayer(nn.Module):
    def __init__(self, gene_list, num_hidden, norm, activation='gelu', dropout=0.3, pe_type=None, cat_pe=True, gene_emb=None, inject_covariate=False, batch_num=None):
        super().__init__()
        self.pe_type = pe_type
        self.cat_pe = cat_pe
        self.act = nn.ReLU()  # create_activation(activation)
        self.norm0 = create_norm(norm, num_hidden)
        self.dropout = nn.Dropout(dropout)
        if pe_type is not None:
            num_hidden_pe = gene_emb.shape[-1] if gene_emb is not None else num_hidden
            if cat_pe:
                num_emb = num_hidden_pe // 2
            else:
                num_emb = num_hidden_pe
            self.pe_enc = select_pe_encoder(pe_type)(num_emb)
        else:
            self.pe_enc = None
            num_emb = num_hidden

        self.feat_enc = OmicsEmbedder(gene_list, gene_emb, num_hidden)

        if inject_covariate:
            self.cov_enc = nn.Embedding(batch_num, num_emb)
            self.inject_covariate = True
        else:
            self.inject_covariate = False

        self.extra_linear = nn.Sequential(
            nn.Linear(self.feat_enc.bb_gene_emb.shape[-1], num_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            create_norm(norm, num_hidden),
        )

    def forward(self, x_dict, input_gene_list=None):
        x = self.feat_enc(x_dict, input_gene_list)  # self.act(self.feat_enc(x_dict, input_gene_list))
        if self.pe_enc is not None:
            pe_input = x_dict[self.pe_enc.pe_key]
            pe = self.pe_enc(pe_input)
            if self.inject_covariate:
                pe = pe + self.cov_enc(x_dict['batch'])
            if self.cat_pe:
                x = torch.cat([x, pe], 1)
            else:
                x = x + pe
        x = self.extra_linear(x)
        return x
