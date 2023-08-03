import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import retrieval_average_precision

import pytorch_lightning as pl
from tqdm import tqdm
from src.clip import clip
from experiments.options import opts


def freeze_model(m):
    m.requires_grad_(False)

def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        elif hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)
        else:
            print(m)

def freeze_all_but_bn2(m):
    for name, param in m.named_parameters():
        if "ln" not in name and "projection" not in name:
            param.requires_grad_(False)

def precak(sim, str_sim, k=None):
    act_lists = [np.nonzero(s)[0] for s in str_sim]
    if type(sim)==torch.Tensor and sim.device!='cpu':
        sim = sim.cpu()
    pred_lists = np.argsort(-sim, axis=1)
    nq = len(act_lists)
    preck_list = []
    reck_list = []
    for iq in range(nq):
        preck_list.append(prec(act_lists[iq], pred_lists[iq], k))
        reck_list.append(rec(act_lists[iq], pred_lists[iq], k))
    return np.mean(preck_list), np.mean(reck_list)


def prec(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    if k is not None:
        pr = len(act_set & pred_set) / min(k, len(pred_set))
    else:
        pr = len(act_set & pred_set) / max(len(pred_set), 1)
    return pr

def rec(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    re = len(act_set & pred_set) / max(len(act_set), 1)
    return re
       
def apsak(sim, str_sim, k=None):
    idx = (-sim).argsort()[:, :k] # MX_batch_size*k, top-k의 idx 나열
    sim_k = np.array([sim[i, id] for i, id in enumerate(idx)]) # MX_batch_size*k, top-k value 나열
    str_sim_k = np.array([str_sim[i, id] for i, id in enumerate(idx)]) # MX_batch_size*k, top-k retrieved value의 target
    
    idx_nz = np.where(str_sim_k.sum(axis=1) != 0)[0] # relevant elements가 있는 idx만 추출
    sim_k = sim_k[idx_nz] # non-zero_batch_size * k
    str_sim_k = str_sim_k[idx_nz] # non-zero_batch_size * k
    
    aps_ = np.zeros((sim.shape[0]), dtype=float) # MX_batch_size * k
    aps_[idx_nz] = aps(sim_k, str_sim_k)
    return aps_


def aps(sim, str_sim):
    sim = torch.tensor(sim); str_sim = torch.BoolTensor(str_sim)
    nq = str_sim.shape[0]
    aps_list = []
    for iq in range(nq):
        aps_list.append(retrieval_average_precision(sim[iq], str_sim[iq])) # query마다 AP 계산
        # aps_list.append(average_precision_score(str_sim[iq], sim[iq])) # query마다 AP 계산
    return aps_list
            
class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.opts = opts
        self.clip, _ = clip.load('ViT-B/32', device=self.device)
        # self.clip.apply(freeze_all_but_bn2)
        # self.clip, _ = clip.load('ViT-L/14', device=self.device)
        freeze_all_but_bn2(self.clip)
        freeze_model(self.clip.transformer)

        # Prompt Engineering
        self.sk_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        self.img_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))

        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=self.distance_fn, margin=self.opts.margin)

        self.best_metric = -1e3

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.clip.parameters(), 'lr': self.opts.clip_LN_lr},
            {'params': [self.sk_prompt] + [self.img_prompt], 'lr': self.opts.prompt_lr}])
        return optimizer

    def forward(self, data, dtype='image'):
        # print("====== ", data.shape, self.img_prompt.shape, self.sk_prompt.shape)
        if dtype == 'image':
            feat = self.clip.encode_image(
                data, self.img_prompt.expand(data.shape[0], -1, -1))
        else:
            feat = self.clip.encode_sketch(
                data, self.sk_prompt.expand(data.shape[0], -1, -1))
        return feat

    def training_step(self, batch, batch_idx):
        sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # sk_tensor, img_tensor, neg_tensor, category = batch[:4]
        sk_tensor, img_tensor, neg_tensor, category, photo_category = batch[:5]
        img_feat = self.forward(img_tensor, dtype='image')
        sk_feat = self.forward(sk_tensor, dtype='sketch')
        neg_feat = self.forward(neg_tensor, dtype='image')

        loss = self.loss_fn(sk_feat, img_feat, neg_feat)
        self.log('val_loss', loss)
        return sk_feat, img_feat, category, photo_category

    def validation_epoch_end(self, val_step_outputs):
        # return
        Len = len(val_step_outputs)
        if Len == 0: return
        query_feat_all = torch.cat([val_step_outputs[i][0] for i in range(Len)]) # 
        gallery_feat_all = torch.cat([val_step_outputs[i][1] for i in range(Len)])
        all_category = np.array(sum([list(val_step_outputs[i][2]) for i in range(Len)], []))
        photo_category = np.array(sum([list(val_step_outputs[i][3]) for i in range(Len)], []))

        if self.opts.instance_level:
            rank = torch.zeros(len(query_feat_all))
            for idx, query_feat in enumerate(query_feat_all):
                dist = self.distance_fn(query_feat.unsqueeze(0), gallery_feat_all)
                trgt_dist = self.distance_fn(
                    query_feat.unsqueeze(0), gallery_feat_all[idx].unsqueeze(0))
                rank[idx] = dist.le(trgt_dist).sum()

            top1 = rank.le(1).sum() / rank.shape[0]
            top10 = rank.le(10).sum() / rank.shape[0]
            meanK = rank.mean()

            self.log('top1', top1)
            self.log('top10', top10)
            self.log('meanK', meanK)

            print ('Metrics:\nTop1: {}, Top10: {}, MeanK: {}'.format(
                top1.item(), top10.item(), meanK.item()
            ))

        else:
            ## mAP category-level SBIR Metrics
            # gallery = gallery_feat_all
            # str_sim = torch.zeros(query_feat_all.size(0), gallery.size(0))
            # sim_euc = torch.zeros(query_feat_all.size(0), gallery.size(0))
            # print("===[INFO]=== ", str_sim.shape)
            # for idx, sk_feat in enumerate(query_feat_all):
            #     category = all_category[idx]
            #     distance = -1*self.distance_fn(sk_feat.unsqueeze(0), gallery) # SOTA
            #     # distance = torch.exp(-torch.cdist(sk_feat.unsqueeze(0), gallery)).squeeze(0) # OURS
            #     sim_euc[idx] = distance
            #     target = torch.zeros(len(gallery), dtype=torch.bool)
            #     target[np.where(all_category == category)] = True
            #     str_sim[idx] = target

            # str_sim = str_sim.cpu().data.numpy()
            # sim_euc = sim_euc.cpu().data.numpy()
            # apsall = apsak(sim_euc, str_sim)
            # aps200 = apsak(sim_euc, str_sim, k=200)
            # print (f'mAP@all: {np.mean(apsall)}\tmAP@200: {np.mean(aps200)}')
            
            # prec100 = np.mean(precak(sim_euc, str_sim, k=100)[0])
            # prec200 = np.mean(precak(sim_euc, str_sim, k=200)[0])
            # print (f'prec@100: {prec100}\tprec@200: {prec200}')
            return query_feat_all, gallery_feat_all, all_category, photo_category
