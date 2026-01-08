import torch
from torch import nn
import torch.nn.functional as F
from ...utils.pe import select_pe_encoder
from ...utils import create_norm


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


class OmicsEmbeddingLayerPerturb(nn.Module):
    def __init__(self, gene_list, num_hidden, norm, activation='gelu', dropout=0.3, pe_type=None, cat_pe=True, gene_emb=None, inject_covariate=False, batch_num=None, use_perturbation=False, num_perturb_conditions=None, gene_embeddings=None, symbol_to_emb_idx=None, condition_to_id=None, case_insensitive_mapping=None):
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
            
        # Perturbation-specific components
        self.use_perturbation = use_perturbation
        
        if use_perturbation:
            if gene_embeddings is not None and symbol_to_emb_idx is not None and condition_to_id is not None:
                # 使用基因嵌入初始化condition嵌入层
                print("Using gene embeddings to initialize condition embeddings...")
                
                # 创建condition嵌入
                condition_embeddings_list = []
                
                # 创建ID到condition名称的反向映射
                id_to_condition = {v: k for k, v in condition_to_id.items()}
                
                for condition_id in range(num_perturb_conditions):
                    # 根据condition_id获取condition名称
                    condition_name = id_to_condition.get(condition_id, "ctrl")
                    
                    # 解析condition并获取嵌入
                    if condition_name.lower() in ["ctrl", "control"]:
                        emb = torch.zeros(gene_embeddings.shape[1])
                    else:
                        # 解析条件，获取基因列表
                        import re
                        genes = re.split(r'[+\-/\|&]', condition_name)
                        genes = [gene.strip() for gene in genes if gene.strip()]
                        
                        gene_embs = []
                        for gene in genes:
                            # 首先尝试精确匹配
                            emb_idx = symbol_to_emb_idx.get(gene)
                            
                            # 如果精确匹配失败，尝试不区分大小写的匹配
                            if emb_idx is None and case_insensitive_mapping is not None:
                                emb_idx = case_insensitive_mapping.get(gene.lower())
                            
                            if emb_idx is not None:
                                gene_emb = torch.tensor(gene_embeddings[emb_idx])
                                gene_embs.append(gene_emb)
                            else:
                                print(f"Warning: Gene {gene} not found in embeddings")
                                gene_embs.append(torch.zeros(gene_embeddings.shape[1]))
                        
                        if gene_embs:
                            # 组合策略：平均
                            emb = torch.stack(gene_embs).mean(dim=0)
                        else:
                            emb = torch.zeros(gene_embeddings.shape[1])
                    
                    condition_embeddings_list.append(emb)
                
                # 初始化嵌入层
                condition_emb_tensor = torch.stack(condition_embeddings_list)
                
                # 检查维度匹配
                if condition_emb_tensor.shape[1] != num_hidden // 4:
                    # 需要维度转换
                    embedding_projection = nn.Linear(condition_emb_tensor.shape[1], num_hidden // 4)
                    condition_emb_tensor = embedding_projection(condition_emb_tensor)
                
                self.perturb_gene_encoder = nn.Embedding.from_pretrained(
                    condition_emb_tensor, 
                    freeze=False  # 允许梯度更新
                )
                print(f"Initialized condition embeddings with gene embeddings for {num_perturb_conditions} conditions")
                
            else:
                # 原有的简单ID编码
                if num_perturb_conditions is not None:
                    self.perturb_gene_encoder = nn.Embedding(num_perturb_conditions, num_hidden // 4)
                    print(f"Initialized random condition embeddings for {num_perturb_conditions} conditions")
                else:
                    self.perturb_gene_encoder = None
            
            self.perturb_flag_encoder = nn.Embedding(2, num_hidden // 4)  # 0: control, 1: perturbed
            self.perturb_fusion = nn.Linear(num_hidden + num_hidden // 2, num_hidden)  # fuse perturbation info
        else:
            self.perturb_flag_encoder = None
            self.perturb_gene_encoder = None
            self.perturb_fusion = None
           
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
        
        # Add perturbation information if available
        if self.use_perturbation and self.perturb_flag_encoder is not None:
            if 'perturb_flag' in x_dict and 'perturb_gene_id' in x_dict:
                # Encode perturbation information
                perturb_flag_emb = self.perturb_flag_encoder(x_dict['perturb_flag'])  # [B, num_hidden//4]
                perturb_gene_emb = self.perturb_gene_encoder(x_dict['perturb_gene_id'])  # [B, num_hidden//4]
                
                # Concatenate perturbation embeddings
                perturb_emb = torch.cat([perturb_flag_emb, perturb_gene_emb], dim=-1)  # [B, num_hidden//2]
                
                # Fuse with gene expression features
                combined_features = torch.cat([x, perturb_emb], dim=-1)  # [B, num_hidden + num_hidden//2]
                x = self.perturb_fusion(combined_features)  # [B, num_hidden]
                
        return x
