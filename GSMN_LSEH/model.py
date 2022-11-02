
import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
# from torch.nn.utils.clip_grad import clip_grad_norm
from torch.nn.utils import clip_grad_norm_
import numpy as np
from collections import OrderedDict
from graph_model import VisualGraph, TextualGraph


def torch_cosine_sim(a, b):
    # sc = torch.randn(a.size(0), b.size(0))
    c = a.mm(b.t())
    d = c.max(1)[0]

    one = torch.ones_like(d)
    d = torch.where(d == 0, one, d)

    sc = (c / d).t()

    if torch.cuda.is_available():
        sc = sc.cuda()

    return sc

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers,
                          batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            # cap_emb = (cap_emb[:, :, :cap_emb.size(2) / 2] +
            #            cap_emb[:, :, cap_emb.size(2) / 2:]) / 2
            cap_emb = (cap_emb[:, :, :int(cap_emb.size(2) / 2)] +
                       cap_emb[:, :, int(cap_emb.size(2) / 2):]) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb, cap_len


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        # print(cost_s.sum() + cost_im.sum())
        return cost_s.sum() + cost_im.sum()

class ContrastiveLossLSEH(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLossLSEH, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation
        self.torchsim = torch_cosine_sim

    def forward(self, scores, ids, img_ids, svd):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        if torch.cuda.is_available():
            svd = svd.cuda()

        ids = np.array(ids)
        img_ids = np.array(img_ids)

        ids = ids // 5
        map = torch.ones(len(ids), len(img_ids))
        for i in range(len(img_ids)):
            for j in range(len(ids)):
                if img_ids[i] == ids[j]:
                    map[i, j] = 0

        if torch.cuda.is_available():
            map = map.cuda()

        lm = 0.05
        SeScores = self.torchsim(svd, svd)
        SeScores = SeScores * lm
        SeMargin = SeScores + self.margin
        SeMargin = SeMargin * map

        # clear diagonals
        maskL = torch.eye(SeMargin.size(0)) > .5
        IL = Variable(maskL)
        if torch.cuda.is_available():
            IL = IL.cuda()
        SeMargin = SeMargin.masked_fill_(IL, 0)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (SeMargin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (SeMargin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

class ContrastiveLossPoly(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLossPoly, self).__init__()

        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def polyloss(self, sim_mat):
        epsilon = 1e-5
        size = sim_mat.size(0)
        hh = sim_mat.t()
        label = torch.Tensor([i for i in range(size)])

        loss = list()
        for i in range(size):
            pos_pair_ = sim_mat[i][i]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][label != label[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]

            pos_pair = pos_pair_
            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            pos_loss = torch.clamp(0.2 * torch.pow(pos_pair, 2) - 0.7 * pos_pair + 0.5, min=0)
            neg_pair = max(neg_pair)
            neg_loss = torch.clamp(0.9 * torch.pow(neg_pair, 2) - 0.4 * neg_pair + 0.03, min=0)

            loss.append(pos_loss + neg_loss)
        for i in range(size):
            pos_pair_ = hh[i][i]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = hh[i][label != label[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]

            pos_pair = pos_pair_
            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue
            pos_loss = torch.clamp(0.2 * torch.pow(pos_pair, 2) - 0.7 * pos_pair + 0.5, min=0)

            neg_pair = max(neg_pair)
            neg_loss = torch.clamp(0.9 * torch.pow(neg_pair, 2) - 0.4 * neg_pair + 0.03, min=0)
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / size
        return loss

    # def forward(self, im, s, s_l):
    def forward(self, scores):
        # compute image-sentence score matrix
        # if self.opt.cross_attn == 't2i':
        #     scores = xattn_score_t2i(im, s, s_l, self.opt)
        # elif self.opt.cross_attn == 'i2t':
        #     scores = xattn_score_i2t(im, s, s_l, self.opt)
        # else:
        #     raise ValueError("unknown first norm type:", self.opt.raw_feature_norm)
        loss = self.polyloss(scores)
        return loss

class GSMN(object):
    """
    Graph Structured Network for Image-Text Matching (GSMN)
    """

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImagePrecomp(
            opt.img_dim, opt.embed_size, opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_bi_gru=opt.bi_gru,
                                   no_txtnorm=opt.no_txtnorm)
        self.i2t_match_G = VisualGraph(
            opt.feat_dim, opt.hid_dim, opt.out_dim, dropout=.5)
        self.t2i_match_G = TextualGraph(
            opt.feat_dim, opt.hid_dim, opt.out_dim, dropout=.5)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.i2t_match_G.cuda()
            self.t2i_match_G.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        # self.criterion = ContrastiveLoss(opt=opt,
        #                                  margin=opt.margin,
        #                                  max_violation=opt.max_violation)

        # self.criterion = ContrastiveLossPoly(opt=opt,
        #                                  margin=opt.margin,
        #                                  max_violation=opt.max_violation)

        # self.criterion = ContrastiveLossE(opt=opt,
        #                                  margin=opt.margin,
        #                                  max_violation=opt.max_violation)

        self.criterion = ContrastiveLossLSEH(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())
        params += list(self.i2t_match_G.parameters())
        params += list(self.t2i_match_G.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0
        self.opt = opt

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(),
                      self.i2t_match_G.state_dict(),
                      self.t2i_match_G.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.i2t_match_G.load_state_dict(state_dict[2])
        self.t2i_match_G.load_state_dict(state_dict[3])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        self.i2t_match_G.train()
        self.t2i_match_G.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        self.i2t_match_G.eval()
        self.t2i_match_G.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        # images = Variable(images, volatile=volatile)
        # captions = Variable(captions, volatile=volatile)

        with torch.no_grad():
            images = Variable(images)
        with torch.no_grad():
            captions = Variable(captions)

        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)

        # cap_emb (tensor), cap_lens (list)
        cap_emb, cap_lens = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, cap_lens

    def forward_sim(self, img_emb, cap_emb, bbox, depends, cap_lens):
        i2t_scores = self.i2t_match_G(
            img_emb, cap_emb, bbox, cap_lens, self.opt)
        t2i_scores = self.t2i_match_G(
            img_emb, cap_emb, depends, cap_lens, self.opt)
        scores = i2t_scores + t2i_scores
        return scores

    def forward_loss(self, scores, ids, img_ids, lsa, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        # loss = self.criterion(scores)
        loss = self.criterion(scores, ids, img_ids, lsa)
        self.logger.update('Le', loss.item())
        return loss

    def train_emb(self, images, captions, bboxes, depends, lengths, ids, img_ids, lsa, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, cap_lens = self.forward_emb(
            images, captions, lengths)

        scores = self.forward_sim(img_emb, cap_emb, bboxes, depends, cap_lens)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(scores, ids, img_ids, lsa)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
