import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
import math, copy
import matplotlib.pyplot as plt
import numpy as np
 

from .module import *

def get_loss_weighted(y_pred, y_true, weights):
    y_true = y_true.float()
    weights = weights.float()
    loss_fn = nn.BCELoss(weight=weights)
    return loss_fn(y_pred, y_true)

class SElayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SElayer,self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
            )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class PromptAttention(nn.Module):
    def __init__(self, input_dim, querydim, nhidden=128, promptAttn_dim=64, num_heads=8): 
        super(PromptAttention, self).__init__()
        assert promptAttn_dim % num_heads == 0 
        self.promptAttn_dim = promptAttn_dim 
        self.promptAttn_dim_k = promptAttn_dim // num_heads 
        self.h = num_heads 
        self.dim = input_dim 
        self.nhidden = nhidden 
        self.selayer = SElayer(querydim, reduction=16)
        self.linears = nn.ModuleList([nn.Linear(promptAttn_dim, promptAttn_dim), 
                                      nn.Linear(promptAttn_dim, promptAttn_dim), 
                                      nn.Linear(input_dim*num_heads, nhidden)]) 
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        dim = value.size(-1) # 128
        batch, h, querydim, d_k = query.size()
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        
        if mask is not None:
            if len(mask.shape)==3:
                mask=mask.unsqueeze(-1)
            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -10000)

        attn_weights = F.softmax(scores, dim = -2)
        if dropout is not None:
            attn_weights=F.dropout(attn_weights, p=dropout, training=self.training)
        weighted_values = torch.matmul(attn_weights, value.unsqueeze(-3))
        return weighted_values, attn_weights

    def forward(self, query, key, value, mask=None, dropout=0.1):
        batch, _, dim = value.size() 
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        query, key = [l(x).view(x.size(0), -1, self.h, self.promptAttn_dim_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        
        x, _ = self.attention(query, key, value, mask, dropout) 
        x = x.transpose(1, 2).contiguous() \
             .view(batch, -1, self.h * dim)
        return self.linears[-1](x)
    
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


'''Gate function'''
class gateMLP(nn.Module):
    def __init__(self,input_dim, hidden_size, output_dim, dropout=0.1):
        super().__init__()

        self.gate = nn.Sequential(
             nn.Dropout(dropout),
             nn.Linear(input_dim, hidden_size),
             nn.ReLU(),
             nn.Linear(hidden_size, output_dim),
             nn.Sigmoid()
        )
        self._initialize()

    def _initialize(self):
        for model in [self.gate]:
            for layer in model:
                if type(layer) in [nn.Linear]:
                    torch.nn.init.xavier_normal_(layer.weight)

    def forward(self,hidden_states ):
        gate_logits = self.gate(hidden_states)
        return gate_logits    
    

'''Frozen Text Representation'''
class BertForRepresentation(nn.Module):
    def __init__(self, args, BioBert):
        super().__init__()
        self.bert = BioBert
        self.language_model=args.language_model
        if self.language_model in ['ClinicalBERT']:
            self.dropout = torch.nn.Dropout(BioBert.config.dropout)
        else:
            self.dropout = torch.nn.Dropout(BioBert.config.hidden_dropout_prob)

    def forward(self, input_ids_sequence, attention_mask_sequence):
        txt_arr = []
        for input_ids, attention_mask  in zip(input_ids_sequence, attention_mask_sequence):
            text_embeddings=self.bert(input_ids, attention_mask=attention_mask)
            text_embeddings= text_embeddings[0][:,0,:]
            text_embeddings = self.dropout(text_embeddings)
            txt_arr.append(text_embeddings)
        return torch.stack(txt_arr)
    

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


'''CLEAR Network'''
class CLEAR(nn.Module):
    def __init__(self, args, triage_dim, numerical_sequence_parameters, category_sequence_parameters1, category_sequence_parameters2, \
                 text_dim, embed_dim, hidden_dim, device, output_dim=2, BioBERT=None):
        super(CLEAR, self).__init__()
        self.triage_dim = triage_dim
        self.numerical_sequence_parameters = numerical_sequence_parameters
        self.text_dim = text_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.kernel_size = 1
        
        self.promptAttn_dim = 31
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.layers = args.layers
        self.task = args.task
        self.beta = 0.1 # 

        self.device = device

        self.relu=nn.ReLU() 
        self.tanh=nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.gate_categ = gateMLP(embed_dim * 2, embed_dim, 1, dropout=0.1)
        self.gate_numer = gateMLP(embed_dim * 2, embed_dim, 1, dropout=0.1)
        self.gate_med = gateMLP(embed_dim * 2, embed_dim, 1, dropout=0.1)
        self.gate_diag = gateMLP(embed_dim * 2, embed_dim, 1, dropout=0.1)
        self.gate_txt = gateMLP(embed_dim * 2, embed_dim, 1, dropout=0.1)
        self.mhsa_c = MultiHeadedAttention(h=8, d_model = embed_dim, dropout=0.1)
        self.mhsa_n = MultiHeadedAttention(h=8, d_model = embed_dim, dropout=0.1)
        self.mhsa_t = MultiHeadedAttention(h=8, d_model = embed_dim, dropout=0.1)
        self.mhsa_fusion = MultiHeadedAttention(h=1, d_model = embed_dim * 2, dropout=0.1)
        self.SublayerConnection = SublayerConnection(self.embed_dim, dropout = self.dropout)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim * 2)
        self.DropoutLayer = nn.Dropout(self.dropout)

        if 'category_attributes' in args.modalities:
            self.proj_category = nn.Conv1d(self.triage_dim, self.embed_dim, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)
            self.hardsamp_linear_category = nn.Linear(self.embed_dim, self.embed_dim)
            
        if 'numerical_sequence' in args.modalities:
            self.AdaptEmbedding_numericalseq = nn.GRU(
            input_size=self.numerical_sequence_parameters['input_dim'],
            hidden_size=self.numerical_sequence_parameters['rnn_dim'],
            num_layers=1,
            bidirectional=True,
            batch_first=True
            )
            self.numerical_sequence_linear = nn.Linear(self.numerical_sequence_parameters['rnn_dim']*2, self.embed_dim)

        if 'notes' in args.modalities:    
            self.proj_txt = nn.Conv1d(self.text_dim, self.embed_dim, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)
            self.hardsamp_linear_text = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)
        
        if 'category_sequence1' in args.modalities:
            self.med_category_sequence_linear1 = nn.Linear(category_sequence_parameters1['input_dim'], category_sequence_parameters1['embed_dim'])
            self.med_AdaptEmbedding_categoryseq = nn.GRU(
            input_size=category_sequence_parameters1['embed_dim'],
            hidden_size=category_sequence_parameters1['rnn_dim'],
            num_layers=1,
            bidirectional=True,
            batch_first=True
            )
            self.med_category_sequence_linear2 = nn.Linear(category_sequence_parameters1['rnn_dim']*2, self.embed_dim)

        if 'category_sequence2' in args.modalities:
            self.diag_category_sequence_linear1 = nn.Linear(category_sequence_parameters2['input_dim'], category_sequence_parameters2['embed_dim'])
            self.diag_AdaptEmbedding_categoryseq = nn.GRU(
            input_size=category_sequence_parameters2['embed_dim'],
            hidden_size=category_sequence_parameters2['rnn_dim'],
            num_layers=1,
            bidirectional=True,
            batch_first=True
            )
            self.diag_category_sequence_linear2 = nn.Linear(category_sequence_parameters2['rnn_dim']*2, self.embed_dim)
        
        self.querylinear = nn.Linear(768, 128)
        self.mquerylinear = nn.Linear(768, 128)

        self.keyprojection_categ = nn.Linear(1, self.promptAttn_dim)
        self.keylinear_categ = nn.Linear(1, 1)

        self.keylinear1_numers = nn.Linear(args.num_of_labtests, 1)
        self.keyprojection_numers = nn.Linear(1, self.promptAttn_dim)
        self.keylinear2_numers = nn.Linear(1, 1)
        self.keylinear3_numers = nn.Linear(args.num_of_labtests, 1)
        self.keylinear4_numers = nn.Linear(args.num_of_labtests, 1)

        self.keylinear1_med = nn.Linear(category_sequence_parameters1['input_dim'], self.embed_dim)
        self.keyprojection_med = nn.Linear(1, self.promptAttn_dim)
        self.keylinear2_med = nn.Linear(1, 1)
        self.keylinear3_med = nn.Linear(category_sequence_parameters1['input_dim'], self.embed_dim)

        self.keylinear1_diag = nn.Linear(category_sequence_parameters2['input_dim'], self.embed_dim)
        self.keyprojection_diag = nn.Linear(1, self.promptAttn_dim)
        self.keylinear2_diag = nn.Linear(1, 1)
        self.keylinear3_diag = nn.Linear(category_sequence_parameters2['input_dim'], self.embed_dim)

        self.keylinear1_txt = nn.Linear(args.num_of_notes, 1)
        self.keyprojection_txt = nn.Linear(1, self.promptAttn_dim)
        self.keylinear2_txt = nn.Linear(1, 1)
        self.keylinear3_txt = nn.Linear(args.num_of_notes, 1)
        self.keylinear4_txt = nn.Linear(args.num_of_notes, 1)
        

        self.keylinear5_categ = nn.Linear(1, 64)
        self.keylinear5_numers = nn.Linear(1, 64)
        self.keylinear5_med = nn.Linear(1, 64)
        self.keylinear5_diag = nn.Linear(1, 64)
        self.keylinear5_txt = nn.Linear(1, 64)
        

        self.keyprojection_text = nn.Linear(1, self.promptAttn_dim)
        self.tt_linear = nn.Linear(self.text_dim, self.embed_dim)
        self.mt_linear = nn.Linear(self.text_dim, self.embed_dim)
        
        self.prompt_attention_triage = PromptAttention(input_dim = self.embed_dim, 
                                                       querydim = 128,
                                                       nhidden = self.embed_dim,
                                                       promptAttn_dim = 128, 
                                                       num_heads = 8)
        
        self.prompt_attention_numerical = PromptAttention(input_dim = self.embed_dim,
                                                          querydim = 128, 
                                                          nhidden = self.embed_dim,
                                                          promptAttn_dim = 128, 
                                                          num_heads = 8)
        
        self.prompt_attention_med = PromptAttention(input_dim = self.embed_dim,
                                                          querydim = 128, 
                                                          nhidden = self.embed_dim,
                                                          promptAttn_dim = 128, 
                                                          num_heads = 8)
        
        self.prompt_attention_diag = PromptAttention(input_dim = self.embed_dim,
                                                          querydim = 128, 
                                                          nhidden = self.embed_dim,
                                                          promptAttn_dim = 128, 
                                                          num_heads = 8)

        self.prompt_attention_txt = PromptAttention(input_dim = self.embed_dim,
                                                          querydim = 128, 
                                                          nhidden = self.embed_dim,
                                                          promptAttn_dim = 128, 
                                                          num_heads = 8)
        
 
        self.intra_modality_self_attention = self.get_cross_network(layers=args.cross_layers)
        self.inter_modality_representation_interaction = TransformerRepresentationInteraction(
                                                                                    embed_dim = 128,
                                                                                    num_heads = 8,
                                                                                    attn_dropout = 0.1,
                                                                                    res_dropout = 0.1,
                                                                                    attn_mask=False
                                                                                    )
        
        disparity_output_size = output_dim if self.task == 'odir_paper' else 2
        self.classifier_disparity = Classifier(input_size = 1280, \
                                       hidden_size = 512, \
                                       output_size = disparity_output_size, \
                                       dropout = 0.1)
        
        self.FFN = PositionwiseFeedForward(128, 256, self.dropout)

        self.proj1 = nn.Linear(self.hidden_dim, self.hidden_dim) 
        self.proj2 = nn.Linear(self.hidden_dim, self.hidden_dim) 
        self.out_layer = nn.Linear(self.hidden_dim, output_dim)

        self.hard_sample_proj1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.hard_sample_proj2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_layer2 = nn.Linear(self.hidden_dim, output_dim)     
        
        self.out_layer0 = nn.Linear(self.hidden_dim, output_dim)
        
        self.addlinear = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU()
        )
        


    def get_cross_network(self, layers=2):
        return TransformerSelfattentionEncoder(embed_dim=128,
                                                num_heads=8,
                                                layers=layers,
                                                attn_dropout=self.dropout,
                                                res_dropout=self.dropout,
                                                attn_mask=False
                                                )

        
    def forward(self, triage_variables, \
                        labtest, \
                        medication_one_hot_tensors, \
                        diagnoses_one_hot_tensors, \
                        text_representations, \
                        task_template, \
                        mask_template, \
                        tao, \
                        mode):
    
    
        category_attributes = triage_variables
        numerical_sequence = labtest
        category_sequence1 = medication_one_hot_tensors
        category_sequence2 = diagnoses_one_hot_tensors
        text_representations = text_representations
        task_template = task_template
        mask_template = mask_template
        
        '''Modality-specific Representation Modeling'''
        category_attributes = category_attributes.unsqueeze(1)
        category_attributes = category_attributes.transpose(1, 2)
        category_attributes = self.proj_category(category_attributes)
        category_attributes = category_attributes.squeeze(dim=-1)
        
        numerical_sequence = self.AdaptEmbedding_numericalseq(numerical_sequence)[0]
        numerical_sequence = self.numerical_sequence_linear(numerical_sequence)
        numerical_sequence = self.tanh(numerical_sequence)      

        category_sequence1 = F.relu(self.med_category_sequence_linear1(category_sequence1))
        category_sequence1 = self.med_AdaptEmbedding_categoryseq(category_sequence1)[0]
        category_sequence1 = self.med_category_sequence_linear2(category_sequence1)
        category_sequence1 = self.tanh(category_sequence1)

        category_sequence2 = F.relu(self.diag_category_sequence_linear1(category_sequence2))
        category_sequence2 = self.diag_AdaptEmbedding_categoryseq(category_sequence2)[0]
        category_sequence2 = self.diag_category_sequence_linear2(category_sequence2)
        category_sequence2 = self.tanh(category_sequence2)

        text_representations1 = text_representations.transpose(1, 2)
        text_representations2 = self.proj_txt(text_representations1)
        text_representations3 = text_representations2.transpose(1, 2)
               
        
        '''Prompt-based Disparity Learning'''        
        '''Counterfactual Prompt Learning'''
        query = self.querylinear(task_template) 
        query_dim = query.size(1)
        mquery = self.mquerylinear(mask_template.unsqueeze(1)) 
        category_attributes0 = category_attributes.unsqueeze(-2) 
        project_query2categ = self.prompt_attention_triage(query, category_attributes0, category_attributes0)
        project_mquery2categ = self.prompt_attention_triage(mquery, category_attributes0, category_attributes0)
        project_mquery2categ2 = project_mquery2categ.repeat_interleave(query_dim, dim=1)
        
        
        project_query2numerical = self.prompt_attention_numerical(query, numerical_sequence, numerical_sequence)
        project_mquery2numerical = self.prompt_attention_numerical(mquery, numerical_sequence, numerical_sequence)
        project_mquery2numerical2 = project_mquery2numerical.repeat_interleave(query_dim, dim=1)
        
        project_query2med = self.prompt_attention_med(query, category_sequence1, category_sequence1)
        project_mquery2med = self.prompt_attention_med(mquery, category_sequence1, category_sequence1)
        project_mquery2med2 = project_mquery2med.repeat_interleave(query_dim, dim=1)
        
        project_query2diag = self.prompt_attention_diag(query, category_sequence2, category_sequence2)
        project_mquery2diag = self.prompt_attention_diag(mquery, category_sequence2, category_sequence2)
        project_mquery2diag2 = project_mquery2diag.repeat_interleave(query_dim, dim=1)
        
        project_query2txt = self.prompt_attention_txt(query, text_representations3, text_representations3)
        project_mquery2txt = self.prompt_attention_txt(mquery, text_representations3, text_representations3)
        project_mquery2txt2 = project_mquery2txt.repeat_interleave(query_dim, dim=1)
        
        enhanced_category_attributes0 = project_query2categ - project_mquery2categ2
        enhanced_numerical_representations0 = project_query2numerical - project_mquery2numerical2
        enhanced_med_representations0 = project_query2med - project_mquery2med2
        enhanced_diag_representations0 = project_query2diag - project_mquery2diag2
        enhanced_txt_representations0 = project_query2txt - project_mquery2txt2
        
        contan_disparity_embedding = torch.cat((enhanced_category_attributes0, 
                                                enhanced_numerical_representations0,
                                                enhanced_med_representations0,
                                                enhanced_diag_representations0,
                                                enhanced_txt_representations0,
                                                ), dim = -1)
        
        contan_disparity_embedding = contan_disparity_embedding.view(contan_disparity_embedding.size(0), -1)
        POP_output = self.classifier_disparity(contan_disparity_embedding)
        
        dispairty = torch.sum(torch.abs(enhanced_category_attributes0)) + \
                    torch.sum(torch.abs(enhanced_numerical_representations0)) + \
                    torch.sum(torch.abs(enhanced_med_representations0)) + \
                    torch.sum(torch.abs(enhanced_diag_representations0)) + \
                    torch.sum(torch.abs(enhanced_txt_representations0)) 
                    
        dispairty = dispairty * 1e-3
        
    
        '''Representation Edit'''
        '''Adaptive Dynamic Imputation'''
        categ_dim = query_dim
        category_attributes = category_attributes.unsqueeze(-2).repeat_interleave(categ_dim, dim=1)
        project_query2categ_prop = self.sigmoid(enhanced_category_attributes0)
        query2categ = torch.where(project_query2categ_prop > tao, category_attributes, torch.tensor(0.0))
        enhanced_category_attributes0 = torch.where(project_query2categ_prop > tao, enhanced_category_attributes0, torch.tensor(0.0))
        
        project_query2numerical_prop = self.sigmoid(enhanced_numerical_representations0)
        query2numerical = torch.where(project_query2numerical_prop > tao, project_query2numerical, torch.tensor(0.0))
        enhanced_numerical_representations0 = torch.where(project_query2numerical_prop > tao, enhanced_numerical_representations0, torch.tensor(0.0))
        
        project_query2med_prop = self.sigmoid(enhanced_med_representations0)
        query2med = torch.where(project_query2med_prop > tao, project_query2med, torch.tensor(0.0))
        enhanced_med_representations0 = torch.where(project_query2med_prop > tao, enhanced_med_representations0, torch.tensor(0.0))
        
        project_query2diag_prop = self.sigmoid(enhanced_diag_representations0)
        query2diag = torch.where(project_query2diag_prop > tao, project_query2diag, torch.tensor(0.0))
        enhanced_diag_representations0 = torch.where(project_query2diag_prop > tao, enhanced_diag_representations0, torch.tensor(0.0))

        project_query2txt_prop = self.sigmoid(enhanced_txt_representations0)
        query2txt = torch.where(project_query2txt_prop > tao, project_query2txt, torch.tensor(0.0))
        enhanced_txt_representations0 = torch.where(project_query2txt_prop > tao, enhanced_txt_representations0, torch.tensor(0.0))
        
        
        '''Soft imputating'''
        merged_category_attributes = torch.cat([query2categ, enhanced_category_attributes0], dim=-1)
        alpha_categ = self.gate_categ(merged_category_attributes)
        enhanced_category_attributes = alpha_categ * query2categ + (1-alpha_categ) * enhanced_category_attributes0
        
        merged_numerical_representations = torch.cat([query2numerical, enhanced_numerical_representations0], dim=-1)
        alpha_numerical = self.gate_numer(merged_numerical_representations)
        enhanced_numerical_representations = alpha_numerical * query2numerical + (1-alpha_numerical) * enhanced_numerical_representations0
        
        merged_med_representations = torch.cat([query2med, enhanced_med_representations0], dim=-1)
        alpha_med = self.gate_med(merged_med_representations)
        enhanced_med_representations = alpha_med * query2med + (1-alpha_med) * enhanced_med_representations0
        
        merged_diag_representations = torch.cat([query2diag, enhanced_diag_representations0], dim=-1)
        alpha_diag = self.gate_diag(merged_diag_representations)
        enhanced_diag_representations = alpha_diag * query2diag + (1-alpha_diag) * enhanced_diag_representations0
        
        merged_lefttxt_representations = torch.cat([query2txt, enhanced_txt_representations0], dim=-1)
        alpha_txt = self.gate_txt(merged_lefttxt_representations)
        enhanced_txt_representations = alpha_txt * query2txt + (1-alpha_txt) * enhanced_txt_representations0
          
        r_category_attributes = F.dropout(enhanced_category_attributes, p=self.dropout, training=self.training)
        r_numerical_representations = F.dropout(enhanced_numerical_representations, p=self.dropout, training=self.training)
        r_med_representations = F.dropout(enhanced_med_representations, p=self.dropout, training=self.training)
        r_diag_representations = F.dropout(enhanced_diag_representations, p=self.dropout, training=self.training)    
        r_text_representations3  = F.dropout(enhanced_txt_representations, p=self.dropout, training=self.training)       
        
        
        '''Modality Fusion'''
        '''Intra-modality Self-attention'''         
        r_category_attributes = r_category_attributes.transpose(0,1)
        r_numerical_representations = r_numerical_representations.transpose(0,1)
        r_med_representations = r_med_representations.transpose(0,1)
        r_diag_representations = r_diag_representations.transpose(0,1)
        r_text_representations3 = r_text_representations3.transpose(0,1)
        self_attention_representions = self.intra_modality_self_attention([r_category_attributes, \
                                                                           r_numerical_representations, \
                                                                           r_med_representations, \
                                                                           r_diag_representations, \
                                                                           r_text_representations3])       
        '''Inter-modality fusion'''
        '''fine-grained discriminative representation interaction '''
        conca_embedding = torch.cat((self_attention_representions[0], 
                                     self_attention_representions[1],
                                     self_attention_representions[2], 
                                     self_attention_representions[3],
                                     self_attention_representions[4],
                                     ), dim = 0)
        conca_embedding = self.inter_modality_representation_interaction(conca_embedding).transpose(0,1)        
        conca_embedding = self.FFN(conca_embedding)
        

        '''Prediction'''        
        conca_embedding = conca_embedding.reshape(conca_embedding.size(0), -1)
        residual_embedding = conca_embedding
        final_embedding = self.proj2(F.dropout(F.relu(self.proj1(conca_embedding)), p=self.dropout, training = self.training))
        final_embedding += residual_embedding
        output = self.out_layer(final_embedding).squeeze(-1)
        
        
        

        if mode == 'train':
            return output, \
                   dispairty, \
                   POP_output

        
        else:
            return torch.sigmoid(output)
