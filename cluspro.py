import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functools import reduce
from operator import mul
from copy import deepcopy
from torch.nn.modules.utils import _pair
from torch.nn.modules.loss import CrossEntropyLoss
from clip_modules.clip_model import load_clip, QuickGELU
from clip_modules.tokenization_clip import SimpleTokenizer
from model.common import *
from .multi_head_attention import CrossAttention
from .nce_loss import *
from sklearn.cluster import KMeans
import random
from .otgcc import *
from .hsic import *

class Adapter(nn.Module):
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="0.1",
                 adapter_layernorm_option="none"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        self.init_option = init_option

        self._reset_parameters()

    def _reset_parameters(self):
        if self.init_option == "bert":
            raise NotImplementedError
        elif self.init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class Disentangler(nn.Module):
    def __init__(self, emb_dim):
        super(Disentangler, self).__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.bn1_fc = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x

class Disentangler2(nn.Module):
    def __init__(self, emb_dim):
        super(Disentangler2, self).__init__()
        self.fc1 = nn.Linear(emb_dim*2, emb_dim)
        self.bn1_fc = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x


class MulitHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads).permute(0,2,1,3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1,):
        super().__init__()
        self.cross_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)
        self.norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, q, kv):
        q = q + self.cross_attn(q, kv, kv)
        q = q + self.dropout(self.mlp(self.norm(q)))
        return q

def distributed_sinkhorn(out, sinkhorn_iterations=3, epsilon=0.05):
    L = torch.exp(out / epsilon).t() # K x B
    B = L.shape[1]
    K = L.shape[0]

    # make the matrix sums to 1
    sum_L = torch.sum(L)
    L /= sum_L

    for _ in range(sinkhorn_iterations):
        L /= torch.sum(L, dim=1, keepdim=True)
        L /= K

        L /= torch.sum(L, dim=0, keepdim=True)
        L /= B

    L *= B
    L = L.t()

    indexs = torch.argmax(L, dim=1)
    # L = torch.nn.functional.one_hot(indexs, num_classes=L.shape[1]).float()
    L = F.gumbel_softmax(L, tau=0.5, hard=True)
    

    return L, indexs
def entropic_COT_gcg(a, b, M, reg1, reg2, f, df, G0=None, numItermax=10,
        numInnerItermax=200, stopThr=1e-9, stopThr2=1e-9, verbose=False, log=False, version='fast'):
    r"""
    modify from ot.optim.gcg in the direction finding part with entropic_partial_wasserstein solver
    ot.optim.gcg: https://pythonot.github.io/_modules/ot/partial.html#partial_gromov_wasserstein

    """
    a, b, M, G0 = list_to_array(a, b, M, G0)
    nx = get_backend(a, M)

    loop = 1

    if log:
        log = {'loss': []}

    if G0 is None:
        G = nx.outer(a, b)
    else:
        G = G0

    def cost(G):
        # t1 = nx.sum(M * G)
        # t2 = nx.sum(G * nx.log(G))
        # t3 = f(G)
        return nx.sum(M * G) + reg1 * nx.sum(G * nx.log(G)) + reg2 * f(G)

    f_val = cost(G)
    if log:
        log['loss'].append(f_val)

    it = 0

    if verbose:
        print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
            'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
        print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, f_val.item(), 0, 0))
    
    if version == 'normal':
        # print("normal")
        func = entropic_COT
    elif version == 'fast':
        func = entropic_COT_fast
        # print("fast")
    while loop:

        it += 1
        old_fval = f_val

        # problem linearization
        Mi = M + reg2 * df(G)

        Gc = func(a, b, Mi, reg1, numInnerItermax)
        if torch.any(torch.isnan(Gc)) or torch.any(torch.isinf(Gc)):
            print('Warning: numerical errors at iteration', it)
            break
        deltaG = Gc - G

        # line search
        dcost = Mi + reg1 * (1 + nx.log(G))  # ??
        alpha, fc, f_val = line_search_armijo(
            cost, G, deltaG, dcost, f_val, alpha_min=0., alpha_max=1.
        )

        G = G + alpha * deltaG

        # test convergence
        if it >= numItermax:
            loop = 0

        abs_delta_fval = abs(f_val - old_fval)
        relative_delta_fval = abs_delta_fval / abs(f_val)

        if relative_delta_fval < stopThr or abs_delta_fval < stopThr2:
            loop = 0

        if log:
            log['loss'].append(f_val)

        if verbose:
            if it % 20 == 0:
                print('{:5s}|{:12s}|{:8s}|{:8s}'.format(
                    'It.', 'Loss', 'Relative loss', 'Absolute loss') + '\n' + '-' * 48)
            print('{:5d}|{:8e}|{:8e}|{:8e}'.format(it, f_val.item(), relative_delta_fval.item(), abs_delta_fval.item()))

    if log:
        return G, log
    else:
        return G

@torch.no_grad()
def entropic_COT_fast(a, b, M, reg, numItermax=1000, stopThr=1e-9, verbose=False, log=False):
        """
        modify from ot.partial.entropic_partial_wasserstein in torch version

        """
        dx = torch.ones_like(a)
        dy = torch.ones_like(b)

        log_e = {'err': []}
        stopThr=1e-9 

        # K = torch.exp(M / (-reg))
        K = M

        Kp = torch.matmul(torch.diag_embed(1 / a, dim1=1), K)
        Kq = torch.matmul(torch.diag_embed(1 / b, dim1=1), K.permute(0, 2, 1))

        err, cpt = 1, 0
        u = dx
        v = dy
        while (cpt < numItermax):

            v0 = v
            temp = torch.div(dx, torch.matmul(Kp, v.unsqueeze(-1)).squeeze(-1))
            u = torch.minimum(temp, dx)
            v = torch.div(dy, torch.matmul(Kq, u.unsqueeze(-1)).squeeze(-1))

            cpt = cpt + 1
            err = (v - v0).abs().mean()
            if err.item() <  stopThr:
                break
        Kprev = torch.matmul(torch.diag_embed(u,dim1=1), K)
        Kprev = torch.matmul(Kprev, torch.diag_embed(v,dim1=1))
        if log:
            return Kprev, log_e
        else:
            return Kprev

def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)

def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)

class cluspro(nn.Module):
    
    def __init__(self, config, attributes, classes, offset):
        super().__init__()
        self.clip = load_clip(name=config.clip_arch, context_length=config.context_length)
        self.tokenizer = SimpleTokenizer()
        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.attr_token = nn.Parameter(torch.zeros(1, len(self.attributes), 768))
        self.obj_token = nn.Parameter(torch.zeros(1, len(self.classes), 768))
        self.attr_dropout = nn.Dropout(config.attr_dropout)
        self.cross_attn_dropout = config.cross_attn_dropout if hasattr(config, 'cross_attn_dropout') else 0
        self.prim_loss_weight = config.prim_loss_weight if hasattr(config, 'prim_loss_weight') else 1

        self.token_ids, self.soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors = self.construct_soft_prompt()
        self.offset = offset
        self.enable_pos_emb = True
        self.nceloss = ContrastiveLoss()
        self.ppcloss = ContrastiveLoss_ppc()
        self.queue_len = 5
        self.cluter_num = 5
        dtype = self.clip.dtype
        if dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.text_encoder = CustomTextEncoder(self.clip, self.tokenizer, self.dtype)
        # freeze CLIP's parameters
        for p in self.parameters():
            p.requires_grad = False

        # only consider ViT as visual encoder
        assert 'ViT' in config.clip_model

        self.additional_visual_params = self.add_visual_tunable_params()

        output_dim = self.clip.visual.output_dim

        self.soft_att_obj = nn.Parameter(self.soft_att_obj)
        self.comp_ctx_vectors = nn.Parameter(comp_ctx_vectors).cuda()
        self.attr_ctx_vectors = nn.Parameter(attr_ctx_vectors).cuda()
        self.obj_ctx_vectors = nn.Parameter(obj_ctx_vectors).cuda()

        self.attr_disentangler = Disentangler(output_dim)
        self.obj_disentangler = Disentangler(output_dim)
        self.attr1 = Disentangler(output_dim)
        self.obj1 = Disentangler(output_dim)

        self.lamda = nn.Parameter(torch.ones(output_dim) * config.init_lamda)
        self.patch_norm = nn.LayerNorm(output_dim)

        self.momentum = 0.99

        #-----------------------set memory-----------------------------
        for i in range(0, len(self.attributes)):
            self.register_buffer("attr_queue" + str(i), torch.zeros(self.cluter_num, 768))
            self.register_buffer("attr_queue_ptr" + str(i), torch.zeros(1, dtype=torch.long))
        
        for i in range(0, len(self.classes)):
            self.register_buffer("obj_queue" + str(i), torch.zeros((self.cluter_num, 768)))
            self.register_buffer("obj_queue_ptr" + str(i), torch.zeros(1, dtype=torch.long))
        
        #-----------------------set feat_cluter-----------------------------
        for i in range(0, len(self.attributes)):
            self.register_buffer("attr_cluter" + str(i), torch.randn(self.cluter_num, 768))
        
        for i in range(0, len(self.classes)):
            self.register_buffer("obj_cluter" + str(i), torch.randn(self.cluter_num, 768))
        
        self.attr_queue = torch.randn((len(self.attributes)*self.cluter_num, 5, 768),requires_grad=False).cuda()
        self.attr_queue = nn.functional.normalize(self.attr_queue, p=2, dim=2)
        self.attr_queue_ptr = torch.zeros(len(self.attributes)*self.cluter_num , dtype=torch.long,requires_grad=False).cuda()
        self.obj_queue = torch.randn((len(self.classes)*self.cluter_num , 5, 768),requires_grad=False).cuda()
        self.obj_queue = nn.functional.normalize(self.obj_queue, p=2, dim=2)
        self.obj_queue_ptr = torch.zeros(len(self.classes)*self.cluter_num, dtype=torch.long,requires_grad=False).cuda()


    def add_visual_tunable_params(self):
        adapter_num = 2 * self.clip.visual.transformer.layers
        params = nn.ModuleList([Adapter(d_model=self.clip.visual.transformer.width, 
                                    bottleneck=self.config.adapter_dim, 
                                    dropout=self.config.adapter_dropout
                                ) for _ in range(adapter_num)])
        return params


    def encode_image(self, x: torch.Tensor):
        return self.encode_image_with_adapter(x)


    def encode_image_with_adapter(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        for i_block in range(self.clip.visual.transformer.layers):
            # MHA
            adapt_x = self.additional_visual_params[i_block](x, add_residual=False)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].attention(
                self.clip.visual.transformer.resblocks[i_block].ln_1(x)
            )
            x = x + adapt_x + residual

            # FFN
            i_adapter = i_block + self.clip.visual.transformer.layers
            adapt_x = self.additional_visual_params[i_adapter](x, add_residual=False)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].mlp(
                self.clip.visual.transformer.resblocks[i_block].ln_2(x)
            )
            x = x + adapt_x + residual
      

        img_feature = x.permute(1, 0, 2)  # LND -> NLD

        img_feature = self.clip.visual.ln_post(img_feature)
        if self.clip.visual.proj is not None:
            img_feature = img_feature @ self.clip.visual.proj
        return img_feature[:, 0, :], img_feature

       


    def encode_text(self, token_ids, token_tensors=None, enable_pos_emb=False):
        return self.text_encoder(token_ids, token_tensors, enable_pos_emb)


    def construct_soft_prompt(self):
        # token_ids indicates the position of [EOS]
        token_ids = self.tokenizer(self.config.prompt_template,
                              context_length=self.config.context_length).cuda()

        tokenized = torch.cat(
            [
                self.tokenizer(tok, context_length=self.config.context_length)
                for tok in self.attributes + self.classes
            ]
        )
        orig_token_embedding = self.clip.token_embedding(tokenized.cuda())
        soft_att_obj = torch.zeros(
            (len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = self.config.ctx_init
        assert isinstance(ctx_init, list)
        n_ctx = [len(ctx.split()) for ctx in ctx_init]
        prompt = self.tokenizer(ctx_init,
                            context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)

        comp_ctx_vectors = embedding[0, 1 : 1 + n_ctx[0], :].to(self.clip.dtype)
        attr_ctx_vectors = embedding[1, 1 : 1 + n_ctx[1], :].to(self.clip.dtype)
        obj_ctx_vectors = embedding[2, 1 : 1 + n_ctx[2], :].to(self.clip.dtype)
        
        return token_ids, soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors


    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        token_tensor, num_elements = list(), [len(pair_idx), self.offset, len(self.classes)]
        for i_element in range(self.token_ids.shape[0]):
            class_token_ids = self.token_ids[i_element].repeat(num_elements[i_element], 1)
            token_tensor.append(self.clip.token_embedding(
                class_token_ids.cuda()
            ).type(self.clip.dtype))

        eos_idx = [int(self.token_ids[i_element].argmax()) for i_element in range(self.token_ids.shape[0])]
        soft_att_obj = self.attr_dropout(self.soft_att_obj)
        # comp
        token_tensor[0][:, eos_idx[0] - 2, :] = soft_att_obj[
            attr_idx
        ].type(self.clip.dtype)
        token_tensor[0][:, eos_idx[0] - 1, :] = soft_att_obj[
            obj_idx + self.offset
        ].type(self.clip.dtype)
        token_tensor[0][
            :, 1 : len(self.comp_ctx_vectors) + 1, :
        ] = self.comp_ctx_vectors.type(self.clip.dtype)
        # attr
        token_tensor[1][:, eos_idx[1] - 1, :] = soft_att_obj[
            :self.offset
        ].type(self.clip.dtype)
        token_tensor[1][
            :, 1 : len(self.attr_ctx_vectors) + 1, :
        ] = self.attr_ctx_vectors.type(self.clip.dtype)
        # obj
        token_tensor[2][:, eos_idx[2] - 1, :] = soft_att_obj[
            self.offset:
        ].type(self.clip.dtype)
        token_tensor[2][
            :, 1 : len(self.obj_ctx_vectors) + 1, :
        ] = self.obj_ctx_vectors.type(self.clip.dtype)

        return token_tensor
    

    def loss_calu(self, predict, target):
        loss_fn = CrossEntropyLoss()
        _, batch_attr, batch_obj, batch_target = target[0],target[1],target[2],target[3]
        if self.training:
            comp_logits, attr_logits, obj_logits, loss_contras,loss_hsic= predict
        else:
            comp_logits, attr_logits, obj_logits = predict
        batch_attr = batch_attr.cuda()
        batch_obj = batch_obj.cuda()
        batch_target = batch_target.cuda()
        loss_comp = loss_fn(comp_logits, batch_target)
        loss_attr = loss_fn(attr_logits, batch_attr)
        loss_obj = loss_fn(obj_logits, batch_obj)
        if self.training:
            loss = loss_comp * self.config.pair_loss_weight +\
                loss_attr * self.config.attr_loss_weight +\
                loss_obj * self.config.obj_loss_weight + 0.1*loss_contras  # +0.1*ppc
        else:
            loss = loss_comp * self.config.pair_loss_weight +\
               loss_attr * self.config.attr_loss_weight +\
               loss_obj * self.config.obj_loss_weight
        return loss


    def logit_infer(self, predict, pairs):
        comp_logits, attr_logits, obj_logits = predict
        attr_pred = F.softmax(attr_logits, dim=-1)
        obj_pred = F.softmax(obj_logits, dim=-1)
        for i_comp in range(comp_logits.shape[-1]):
            weighted_attr_pred = 1 if self.config.attr_inference_weight == 0 else attr_pred[:, pairs[i_comp][0]] * self.config.attr_inference_weight
            weighted_obj_pred = 1 if self.config.obj_inference_weight == 0 else obj_pred[:, pairs[i_comp][1]] * self.config.obj_inference_weight
            comp_logits[:, i_comp] = comp_logits[:, i_comp] * self.config.pair_inference_weight + weighted_attr_pred * weighted_obj_pred
        return comp_logits

    
    def encode_text_for_open(self, idx):
        token_tensors = self.construct_token_tensors(idx)
        text_features = []
        for i_element in range(self.token_ids.shape[0]):
            _text_features, _ = self.encode_text(
                self.token_ids[i_element],
                token_tensors[i_element],
                enable_pos_emb=self.enable_pos_emb,
            )

            idx_text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )
            text_features.append(idx_text_features)
        return text_features

    
    def forward_for_open(self, batch, text_feats):
        batch_img = batch[0].cuda()
        b = batch_img.shape[0]
        # l, _ = idx.shape
        batch_img, batch_patch = self.encode_image(batch_img.type(self.clip.dtype))
        batch_img_features = [batch_img, self.attr_disentangler(batch_img), self.obj_disentangler(batch_img)]
        normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]

        logits = list()
        for i_element in range(self.token_ids.shape[0]):
            idx_text_features = text_feats[i_element]

            # CMT
            cmt_text_features = idx_text_features.unsqueeze(0).expand(b, -1, -1)
            batch_patch = self.patch_norm(batch_patch)
            for layer in self.cmt:
                cmt_text_features = layer(cmt_text_features, batch_patch)
            cmt_text_features = idx_text_features + self.lamda * cmt_text_features.squeeze(1)

            cmt_text_features = cmt_text_features / cmt_text_features.norm(
                dim=-1, keepdim=True
            )

            logits.append(
                torch.einsum(
                    "bd, bkd->bk", 
                    normalized_img_features[i_element], 
                    cmt_text_features * self.clip.logit_scale.exp()
            ))
        return logits

    def pos_neg(self,attr_idx,attr_cls,leixing):
        N = attr_cls.shape[0]  #
        index = attr_idx.reshape(N,1)
        # print(index)
        onehot = torch.zeros(N,len(leixing)*self.cluter_num)
        onehot.scatter_(1, index, 1)  # 8,n,c
        #print("one",onehot)
        onehot = onehot.bool()
        onehot_fan = torch.logical_not(onehot)
        attr_cls = attr_cls.repeat(N,1,1,1)
        #print("onehot",onehot.shape,attr_cls.shape)
        #print(onehot)
        attr_cls_pos = attr_cls[onehot,:,:].reshape(N,-1,768)
        attr_cls_neg = attr_cls[onehot_fan,:,:].reshape(N,-1,768)
        #print("asdff", attr_cls_neg.shape)
        return attr_cls_pos, attr_cls_neg


    @torch.no_grad()
    def _dequeue_and_enqueue(self, x, label, leixing):
        if leixing =="attr":
           queue_ptr_i = self.attr_queue_ptr[label]
           ptr = int(queue_ptr_i)
           self.attr_queue[label,ptr,:] = x.detach()
           ptr = (ptr + 1) % self.queue_len  # move pointer
           self.attr_queue_ptr[label] = ptr
           #print("attr_queue",self.attr_queue[label,ptr,:].shape)
        else:
           queue_ptr_i = self.obj_queue_ptr[label]
           ptr = int(queue_ptr_i)
           self.obj_queue[label,ptr,:] = x.detach()
           ptr = (ptr + 1) % self.queue_len  # move pointer
           self.obj_queue_ptr[label] = ptr
           #print("obj_queue",self.obj_queue[label,ptr,:].shape)
            
       
    def _sample_negative(self,attr_idx,attr_cls,leixing):
        N = attr_idx.shape[0]  #
        index = attr_idx.reshape(N,1)
        # print(index)
        onehot = torch.zeros(N,len(leixing)*self.cluter_num).cuda()
        onehot.scatter_(1, index, 1)  # 8,n,c
        #print("one",onehot)
        onehot = onehot.bool()
        onehot_fan = torch.logical_not(onehot)
        #print("onehot",onehot.shape,attr_cls.shape)
        #print(onehot)
        attr_cls_pos = attr_cls[onehot,:,:].reshape(N,-1,768)
        attr_cls_neg = attr_cls[onehot_fan,:,:].reshape(N,-1,768)
        #print("asdff", attr_cls_neg.shape)
        return attr_cls_pos, attr_cls_neg
            

    def train_forward(self, batch, idx):
        batch_img= batch[0].cuda()
        
        attr_idx, obj_idx = batch[1],batch[2]
        batch_img, batch_patch = self.encode_image(batch_img.type(self.clip.dtype))
        batch_num = batch_img.shape[0]
        batch_attr = self.attr_disentangler(batch_img) #b,p,768
        batch_obj = self.obj_disentangler(batch_img)  #8,256,76
        results = []
        results1 = []
        featattr_memory = getattr(self, "attr_queue0")
        results.append(featattr_memory)
 
        for k in range(1, len(self.attributes)):
           
            featattr_memory = torch.cat((featattr_memory, getattr(self, "attr_queue" + str(k))), 0) #
            results.append(getattr(self, "attr_queue" + str(k)))
        
        featobj_memory = getattr(self, "obj_queue0")
        results1.append(featobj_memory)
 
        for k in range(1, len(self.classes)):
   
            featobj_memory = torch.cat((featobj_memory, getattr(self, "obj_queue" + str(k))), 0) # n,10,c
            results1.append(getattr(self, "obj_queue" + str(k)))

     

        attr_masks = torch.einsum('bd,kmd->bmk', l2_normalize(batch_attr), featattr_memory.detach().reshape(-1,self.cluter_num,768))   # n,m,c
        obj_masks = torch.einsum('bd,kmd->bmk', l2_normalize(batch_obj), featobj_memory.detach().reshape(-1,self.cluter_num,768))   # n,m,c

        attr_labels = torch.zeros(batch_num, dtype=torch.long).cuda()
        obj_labels = torch.zeros(batch_num, dtype=torch.long).cuda()
   
        with torch.no_grad():
            for k in range(len(self.attributes)):
                init_q = attr_masks[...,k]  
                couplings, selected_mask  = local_assign(batch_attr.detach(), init_q.detach(), top_percent=1)
         
                
                
                
                couplings = couplings.float()
         
                indexs = torch.argmax(couplings, dim=1)
                q = F.gumbel_softmax(couplings, tau=0.5, hard=True)  #
                q = q[attr_idx==k,...] 
                c_k = batch_attr[attr_idx==k, ...]
                if init_q.shape[0] == 0:
                    continue
                
           
                f = q.permute(1,0) @ c_k
                queue_attr = getattr(self, "attr"+"_queue" + str(int(k)))
           
                f= F.normalize(f, p=2, dim=-1)
                n = torch.sum(q, dim=0)
             
                queue_attr[n != 0,:]= queue_attr[n != 0,:] * self.momentum + f[n != 0,:]* (1 - self.momentum)

                queue_attr = l2_normalize(queue_attr)
                attr_labels[attr_idx==k] = indexs[attr_idx==k].long() + (self.cluter_num* k)
                

            for k in range(len(self.classes)):
                init_q = obj_masks[...,k]  
            
                couplings, selected_mask  = local_assign(batch_obj.detach(), init_q.detach(), top_percent=1)
                
                couplings = couplings.float()
                indexs = torch.argmax(couplings, dim=1)
                q = F.gumbel_softmax(couplings, tau=0.5, hard=True)
                q = q[obj_idx==k,...] 
        
                c_k = batch_obj[obj_idx==k, ...]
                if init_q.shape[0] == 0:
                    continue
             
      
            
                f = q.permute(1,0) @ c_k
                queue_obj = getattr(self, "obj"+"_queue" + str(int(k)))
                f= F.normalize(f, p=2, dim=-1)
                n = torch.sum(q, dim=0)
                queue_obj[n != 0,:]= queue_obj[n != 0,:] * self.momentum + f[n != 0,:]* (1 - self.momentum)
                queue_obj = l2_normalize(queue_obj)
                obj_labels[obj_idx==k] = indexs[obj_idx==k].long() + (self.cluter_num * k)



        loss_contrastive = 0
        loss_contrastive_ppc = 0
    
        
        featattr_memory_con = torch.stack(results,dim=0)
        featobj_memory_con = torch.stack(results1,dim=0)
        featattr_memory_con = featattr_memory_con.detach().unsqueeze(0).repeat(batch_num,1,1,1) # batch, attr, 5,768
        featobj_memory_con = featobj_memory_con.detach().unsqueeze(0).repeat(batch_num,1,1,1) # batch, obj, 5,768
    
        loss_hsic = 0
        batch_img_features = [batch_img, self.attr1(batch_attr), self.obj1(batch_obj)]
        normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]
       
        attrN = attr_labels.shape[0]  #
        attrindex = attr_labels.reshape(attrN,1)

        attronehot = torch.zeros(attrN,len(self.attributes)*self.cluter_num).cuda()
        attronehot.scatter_(1, attrindex, 1)  # 8,n,c
      
        attronehot = attronehot.bool()
        


        objN = obj_labels.shape[0]  #
        objindex = obj_labels.reshape(objN,1)
 
        objonehot = torch.zeros(objN,len(self.classes)*self.cluter_num).cuda()
        objonehot.scatter_(1, objindex, 1)  # 8,n,c
    
        objonehot = objonehot.bool()
 
        token_tensors = self.construct_token_tensors(idx)
       
        attr_queue = self.attr_queue.detach().unsqueeze(0).repeat(batch_num,1,1,1)
        obj_queue = self.obj_queue.detach().unsqueeze(0).repeat(batch_num,1,1,1)
         
        featattr_memory_con = self.attr1(featattr_memory_con.reshape(-1,768)).reshape(batch_num,-1,768)
        featobj_memory_con = self.obj1(featobj_memory_con.reshape(-1,768)).reshape(batch_num,-1,768)
        attr_queue = self.attr1(attr_queue.reshape(-1,768)).reshape(batch_num,-1,5,768)
        obj_queue = self.obj1(obj_queue.reshape(-1,768)).reshape(batch_num,-1,5,768)
        select_attrpro = featattr_memory_con[attronehot,:]
        select_objpro = featobj_memory_con[objonehot,:]
        loss_hsic1 = hsic_normalized(batch_img_features[1], select_objpro)
        loss_hsic2 =  0.5 *hsic_normalized(batch_img_features[2], select_attrpro)
        loss_hsic = loss_hsic1+loss_hsic2
  
        featattr_memory_con1 =  torch.cat([featattr_memory_con,featobj_memory_con],dim=1)
        featobj_memory_con1 =  torch.cat([featobj_memory_con,featattr_memory_con],dim=1)
      
        loss_contrastive = self.nceloss(batch_img_features[1],featattr_memory_con1,attr_labels) + self.nceloss(batch_img_features[2],featobj_memory_con1,obj_labels)
        
        for i in range(batch_num):
            self._dequeue_and_enqueue(batch_attr[i],attr_labels[i],"attr")
            self._dequeue_and_enqueue(batch_obj[i],obj_labels[i],"obj")

        logits = list()
        text_feature = []
        for i_element in range(self.token_ids.shape[0]):
            
            _text_features, _ = self.encode_text(
                self.token_ids[i_element],
                token_tensors[i_element],
                enable_pos_emb=self.enable_pos_emb,
            )

            idx_text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )

            logits.append(
                torch.einsum(
                    "bd, kd->bk", 
                    normalized_img_features[i_element], 
                    idx_text_features * self.clip.logit_scale.exp()
            ))
        logits.append(loss_contrastive)
        logits.append(loss_hsic)
       
        return logits
    
    def val_forward(self, batch, idx):
        batch_img= batch[0].cuda()
      
        attr_idx, obj_idx = batch[1],batch[2]
  
        

        batch_img, batch_patch = self.encode_image(batch_img.type(self.clip.dtype))
        batch_num = batch_img.shape[0]
   
        batch_attr = self.attr_disentangler(batch_img) #b,p,768
        batch_obj = self.obj_disentangler(batch_img)  #8,256,768
        
        image_same_attr_patch_new = batch_attr
        image_same_obj_patch_new = batch_obj 
    

        results = []
        results1 = []
        featattr_memory = getattr(self, "attr_queue0")
        results.append(featattr_memory)
      
         
        for k in range(1, len(self.attributes)):
        
            featattr_memory = torch.cat((featattr_memory, getattr(self, "attr_queue" + str(k))), 0) #
            results.append(getattr(self, "attr_queue" + str(k)))
        
        featobj_memory = getattr(self, "obj_queue0")
        results1.append(featobj_memory)

        for k in range(1, len(self.classes)):

            featobj_memory = torch.cat((featobj_memory, getattr(self, "obj_queue" + str(k))), 0) 
            results1.append(getattr(self, "obj_queue" + str(k)))

        featattr_memory_con = torch.stack(results,dim=0)
        featobj_memory_con = torch.stack(results1,dim=0)
        featattr_memory_con = featattr_memory_con.detach().unsqueeze(0).repeat(batch_num,1,1,1) 
        featobj_memory_con = featobj_memory_con.detach().unsqueeze(0).repeat(batch_num,1,1,1)
        batch_img_features = [batch_img, self.attr1(batch_attr), self.obj1(batch_obj)]
        normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]
        token_tensors = self.construct_token_tensors(idx) #

        logits = list()
        for i_element in range(self.token_ids.shape[0]):
            
            _text_features, _ = self.encode_text(
                self.token_ids[i_element],
                token_tensors[i_element],
                enable_pos_emb=self.enable_pos_emb,
            )

            idx_text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )

            logits.append(
                torch.einsum(
                    "bd, kd->bk", 
                    normalized_img_features[i_element], 
                    idx_text_features * self.clip.logit_scale.exp()
            ))   
        return logits

    def forward(self,batch, idx):
        if self.training:
            logits = self.train_forward(batch, idx)
        else:
            with torch.no_grad():
                logits = self.val_forward(batch, idx)
        return logits