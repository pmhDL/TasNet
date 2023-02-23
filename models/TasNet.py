import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.backbones import Res12, WRN28
from utils.misc import emb_loss, count_acc
from sklearn.cluster import KMeans

def euclidean_metric(query, proto):
    '''
    :param a: query
    :param b: proto
    :return: (num_sample, way)
    '''
    n = query.shape[0]  # num_samples
    m = proto.shape[0]  # way
    query = query.unsqueeze(1).expand(n, m, -1)
    proto = proto.unsqueeze(0).expand(n, m, -1)
    logits = -((query - proto) ** 2).sum(dim=2)
    return logits  # (way, num_samples)


def cosine_metric(query, proto):
    '''
    :param query:  (bs, dim)
    :param proto:  (way, dim)
    :return: (bs, way)
    '''
    q = query.shape[0]  # bs
    p = proto.shape[0]  # way
    que2 = query.unsqueeze(1).expand(q, p, -1)
    pro2 = proto.unsqueeze(0).expand(q, p, -1)
    logit = torch.cosine_similarity(que2, pro2, dim=2)
    return logit  # (bs, way)

def compute_proto_np(feat, label, way):
    '''numpy'''
    feat_proto = np.zeros((way, feat.shape[1]))
    for lb in np.unique(label):
        ds = np.where(label == lb)[0]
        feat_ = feat[ds]
        feat_proto[lb] = np.mean(feat_, axis=0)
    return feat_proto


def updateproto(Xs, ys, cls_center, way):
    """
    transform the cluster labels to the class labels
    """
    proto = compute_proto_np(Xs, ys, way)
    dist = ((proto[:, np.newaxis, :]-cls_center[np.newaxis, :, :])**2).sum(2)
    id = dist.argmin(1)
    feat_proto = np.zeros((way, Xs.shape[1]))
    for i in range(way):
        # feat_proto[i] = lam * cls_center[id[i]] + (1-lam) * proto[i]
        feat_proto[i] = (cls_center[id[i]] + proto[i])/2
    return feat_proto


class OneFC(nn.Module):
    """The class for inner loop."""
    def __init__(self, way, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.way = way
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.way, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(self.way))
        self.vars.append(self.fc1_b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        net = F.linear(input_x, fc1_w, fc1_b)
        return net

    def parameters(self):
        return self.vars


class TasLearner(nn.Module):
    """The class for outer loop."""
    def __init__(self, args, mode='meta', num_cls=64):
        super().__init__()
        self.args = args
        self.mode = mode
        z_dim = 640
        if self.args.dataset == 'cub':
            out_dim = 312
        else:
            out_dim = 300

        if self.args.model_type == 'res12':
            self.encoder = Res12()
        elif self.args.model_type == 'wrn28':
            self.encoder = WRN28()

        if self.mode == 'pre':
            self.pre_fc = nn.Sequential(nn.Linear(640, num_cls))

        elif self.mode == 'concat' or self.mode == 'fusion':
            self.word_learner = OneFC(out_dim, z_dim)
            self.feat_learner = nn.Linear(z_dim, out_dim)

    def forward(self, inp):
        """The function to forward the model."""
        if self.mode == 'pre':
            return self.pretrain_forward(inp)
        elif self.mode == 'proto':
            data_shot, label_shot, data_query = inp
            return self.proto_forward(data_shot, label_shot, data_query)
        elif self.mode == 'concat':
            data_shot, emb_s, label_shot, data_query = inp
            return self.concat_forward(data_shot, emb_s, label_shot, data_query)
        elif self.mode == 'fusion':
            data_shot, label_shot, emb_s, data_query = inp
            return self.fusion_forward(data_shot, label_shot, emb_s, data_query)
        elif self.mode == 'preval':
            data_shot, label_shot, data_query = inp
            return self.preval_forward(data_shot, label_shot, data_query)
        else:
            raise ValueError('Please set the correct mode.')

    def pretrain_forward(self, inp):
        """The function to forward pretrain phase."""
        return self.pre_fc(self.encoder(inp, map=False))

    def preval_forward(self, data_shot, label_shot, data_query):
        """The function to forward meta-validation during pretrain phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """
        query = self.encoder(data_query, map=False)
        embedding_shot = self.encoder(data_shot, map=False)
        support = embedding_shot.view(self.args.shot, self.args.way, -1).transpose(1, 0)  # (way, shot, dim)
        proto = torch.mean(support, dim=1)
        logits_dist = euclidean_metric(query, proto)
        logits_sim = torch.mm(query, F.normalize(proto, p=2, dim=-1).t())
        return logits_dist, logits_sim


    def proto_forward(self, data_shot, label_shot, data_query):
        query = self.encoder(data_query, map=False)
        embedding_shot = self.encoder(data_shot, map=False)
        support = embedding_shot.view(self.args.shot, self.args.way, -1).transpose(1, 0)
        proto = torch.mean(support, dim=1)
        logits_dist = euclidean_metric(query, proto)
        return logits_dist


    def concat_forward(self, data_shot, emb_s, label_shot, data_query):
        # embedding_query = self.encoder(data_query, map=False)
        # embedding_shot = self.encoder(data_shot, map=False)
        embedding_query = data_query
        embedding_shot = data_shot

        optimizer1 = torch.optim.Adam([{'params': self.word_learner.parameters(), 'lr': self.args.gradlr}
                                      ], weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=10, gamma=0.2)
        if self.args.dataset == 'tiered' or self.args.dataset == 'cub':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=50, gamma=0.5)

        self.word_learner.train()
        for i in range(100):
            support_emb = self.word_learner(embedding_shot)
            emb_s = emb_s.type(support_emb.type())
            loss = emb_loss(support_emb, emb_s, self.args)
            optimizer1.zero_grad()
            loss.backward(retain_graph=True)
            optimizer1.step()
            lr_scheduler.step()
        self.word_learner.eval()
        s_que = self.word_learner(embedding_query)

        vis_query = self.feat_learner(embedding_query)
        vis_support = self.feat_learner(embedding_shot)

        if self.args.setting == 'tran':
            Xq = vis_query.cuda().data.cpu().numpy()
            Xs = vis_support.cuda().data.cpu().numpy()
            ys = label_shot.cuda().data.cpu().numpy()
            km = KMeans(n_clusters=self.args.way, max_iter=1000, random_state=100)
            yq_fit = km.fit(Xq)
            clus_center = yq_fit.cluster_centers_
            proto_v = updateproto(Xs, ys, clus_center, self.args.way)
            proto_v = torch.tensor(proto_v).type(embedding_shot.type())
            visual_proto = F.normalize(proto_v, dim=1)
            comb_proto = torch.cat((visual_proto, emb_s[:self.args.way].type(vis_support.dtype)), dim=1)
        else:
            vis_support1 = vis_support.view(self.args.shot, self.args.way, -1).transpose(1, 0)
            visual_proto = torch.mean(vis_support1, dim=1)
            comb_proto = torch.cat((visual_proto, emb_s[:self.args.way].type(vis_support.dtype)), dim=1)

        comb_que = torch.cat((vis_query, s_que), dim=1)
        comb_que = F.normalize(comb_que, dim=1)
        comb_proto = F.normalize(comb_proto, dim=1)

        lg_a = euclidean_metric(comb_que, comb_proto)
        if self.args.dataset == 'cub' or self.args.dataset == 'tiered':
            lg_a = cosine_metric(comb_que, comb_proto)

        return lg_a

    def fusion_forward(self, data_shot, label_shot, emb_s, data_query):
        # embedding_query = self.encoder(data_query, map=False)
        # embedding_shot = self.encoder(data_shot, map=False)
        embedding_query = data_query
        embedding_shot = data_shot

        optimizer1 = torch.optim.Adam([{'params': self.word_learner.parameters(), 'lr': self.args.gradlr}
                                      ], weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=10, gamma=0.2)
        self.word_learner.train()
        for i in range(100):
            support_emb = self.word_learner(embedding_shot)
            emb_s = emb_s.type(support_emb.type())
            loss = emb_loss(support_emb, emb_s, self.args)
            optimizer1.zero_grad()
            loss.backward(retain_graph=True)
            optimizer1.step()
            lr_scheduler.step()
        self.word_learner.eval()
        s_que = self.word_learner(embedding_query)

        v_que = self.feat_learner(embedding_query)
        v_sup = self.feat_learner(embedding_shot)
        lamda = self.args.lamda

        if self.args.setting == 'tran':
            Xq = v_que.cuda().data.cpu().numpy()
            Xs = v_sup.cuda().data.cpu().numpy()
            ys = label_shot.cuda().data.cpu().numpy()
            km = KMeans(n_clusters=self.args.way, max_iter=1000, random_state=100)
            yq_fit = km.fit(Xq)
            clus_center = yq_fit.cluster_centers_
            proto_v = updateproto(Xs, ys, clus_center, self.args.way)
            proto_v = torch.tensor(proto_v).type(embedding_shot.type())
            v_proto = F.normalize(proto_v, dim=1)
            comb_proto = lamda * v_proto + (1-lamda)*emb_s[:self.args.way].type(v_sup.dtype)
        else:
            vis_support1 = v_sup.view(self.args.shot, self.args.way, -1).transpose(1, 0)
            v_proto = torch.mean(vis_support1, dim=1)
            comb_proto = lamda * v_proto + (1-lamda)*emb_s[:self.args.way].type(v_sup.dtype)
        comb_que = lamda * v_que + (1-lamda)*s_que

        lg_a = euclidean_metric(comb_que, comb_proto)
        if self.args.dataset == 'cub':
            lg_a = cosine_metric(comb_que, comb_proto)

        return lg_a