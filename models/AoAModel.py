from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils

from .CaptionModel import CaptionModel
from .AttModel import pack_wrapper, AttModel


class AoAModel(AttModel):
    def __init__(self, opt):
        super(AoAModel, self).__init__(opt)
        self.num_layers = 2

        # self.ctx2att = nn.Linear(self.ctx_dim, 2 * self.att_hid_dim)
        # self.h2att = nn.Linear(self.rnn_size, 2 * self.att_hid_dim)
        # self.alpha_net = nn.Linear(self.att_hid_dim, 1)

        self.aoa_attn = AoA_Module(opt)
        self.core = AoACore(opt)

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first
        p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, p_att_feats, att_masks

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        # hybrid feature
        # fc_feats_ = self.fc_embed(fc_feats)
        # att_feats_ = self.att_embed(att_feats)
        # mean_att_feats = torch.mean(att_feats_, 1)
        # hybrid_feats = torch.cat([fc_feats_, mean_att_feats], 1)

        batch_size = fc_feats.size(0)
        outputs = []
        att_weights = []

        # embed fc and att feats
        # fc_feats = self.fc_embed(fc_feats)
        # att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        state = self.init_hidden(batch_size)

        # Project the attention feats first
        # p_att_feats = self.ctx2att(att_feats)

        # fixed
        # p_fc_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats)
        # p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        for i in range(self.seq_length + 2):
            if i == 0:
                xt = self.img_embed(fc_feats)
            else:
                if self.training and i >= 2 and self.ss_prob > 0.0:  # otherwiste no need to sample
                    sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                    sample_mask = sample_prob < self.ss_prob
                    if sample_mask.sum() == 0:
                        it = seq[:, i - 1].clone()
                    else:
                        sample_ind = sample_mask.nonzero().view(-1)
                        it = seq[:, i - 1].data.clone()
                        # prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                        # it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                        # prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                        # it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                        prob_prev = torch.exp(outputs[-1].data)  # fetch prev distribution: shape Nx(M+1)
                        it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1)[sample_ind])
                else:
                    it = seq[:, i - 1].clone()
                    # break if all the sequences end
                if i >= 2 and seq[:, i - 1].data.sum() == 0:
                    break
                xt = self.embed(it)

            if i >= 1:
                # self.h2att(h) + p_att_feats(att_feats)
                # att_res = self.aoa_attn(h, att_feats, p_att_feats)

                # mean pooling
                mean_att_feats = torch.mean(att_feats, 1)

                att_res = self.aoa_attn(state[0][-1], att_feats, att_masks)
                # att_weights.append(self.aoa_attn.alpha.clone())
                att_weights.append(self.aoa_attn.alpha)

            output, state = self.core(xt, mean_att_feats, state, att_res)
            # output, state = self.core(xt, fc_feats, state, att_res)
            outputs.append(output)

        outputs = torch.cat([_.unsqueeze(1) for _ in outputs[1:]], 1)

        # START of CHANGE
        if self.training:
            return outputs, att_weights
        else:
            return self.log_softmax(outputs), att_weights
        # END of CHANGE

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first
        p_att_feats = self.ctx2att(att_feats)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image individually for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = fc_feats[k:k + 1].expand(beam_size, self.fc_dim)
            tmp_att_feats = att_feats[k:k + 1].expand(*((beam_size,) + att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = p_att_feats[k:k + 1].expand(*((beam_size,) + p_att_feats.size()[1:])).contiguous()
            tmp_att_masks = att_masks[k:k + 1].expand(
                *((beam_size,) + att_masks.size()[1:])).contiguous() if att_masks is not None else None

            for t in range(1):
                if t == 0:
                    xt = self.img_embed(tmp_fc_feats)

                # att_res = self.att(state[0][-1], tmp_att_feats, tmp_p_att_feats, tmp_att_masks)
                att_res = self.aoa_attn(state[0][-1], tmp_att_feats, tmp_p_att_feats, tmp_att_masks)
                output, state = self.core(xt, state, att_res)
                logprobs = F.log_softmax(self.logit(output))

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats,
                                                  tmp_att_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats, att_masks=None, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        # p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        # p_fc_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats)

        trigrams = []  # will be a list of batch_size dictionaries

        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)
        for t in range(self.seq_length + 2):
            if t == 0:
                xt = self.img_embed(fc_feats)
            else:
                if t == 1:  # input <bos>
                    it = fc_feats.data.new(batch_size).long().zero_()
                elif sample_method == 'greedy':
                    it = sample_n.clone()
                    # stop when all finished
                    if t > 1 and it.sum() == 0:
                        break
                else:
                    logprobs_t = logprobs.data.clone()
                    if temperature > 0:
                        if block_trigrams and t >= 3:
                            # Store trigrams generated at last step
                            prev_two_batch = seq[:, t - 3:t - 1]
                            for i in range(batch_size):  # = seq.size(0)
                                prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                                if prev_two in trigrams[i]:
                                    trigrams[i][prev_two].add(seq[i, t - 1].item())
                                else:
                                    trigrams[i][prev_two] = set([seq[i, t - 1].item()])
                        # Sample from temperature-softmax distribution
                        if temperature != 1.0:
                            logprobs_t.div_(temperature)
                        prob_prev = torch.exp(logprobs_t)
                        if block_trigrams and t >= 4:
                            # Before sampling, check if we have seen this trigram ...
                            prev_three_batch = seq[:, t - 4:t - 1]
                            for i in range(batch_size):
                                prev_three = (prev_three_batch[i][0].item(), prev_three_batch[i][1].item(),
                                              prev_three_batch[i][2].item())
                                if prev_three in trigrams[i]:
                                    # If so, block all tokens that we have seen
                                    # (and, implicitly, block all <unk> tokens)
                                    # print('block', i, ','.join([self.ix_to_word[str(w.item())] for w in prev_three]), '=>', ','.join([self.ix_to_word[str(w)] for w in trigrams[i][prev_three]]))
                                    for w in trigrams[i][prev_three]:
                                        prob_prev[i, w] = 0
                                # Also block all <unk> tokens
                                prob_prev[i, 0] = 0
                        it = torch.multinomial(prob_prev, 1)
                        sample_logprobs = logprobs_t.gather(1, it)  # gather the logprobs at sampled positions
                        it = it.view(-1).long()  # and flatten indices for downstream processing

                # if t >= 1:
                #     # stop when all finished
                #     if t > 1 and seq[:,t-2].sum() == 0:
                #         break

                xt = self.embed(it)

            if t >= 1:
                # att_res = self.att(state[0][-1], p_att_feats, pp_att_feats, p_att_masks)
                # mean pooling
                mean_att_feats = torch.mean(att_feats, 1)
                att_res = self.aoa_attn(state[0][-1], att_feats, att_masks)

            output, state = self.core(xt, mean_att_feats, state, att_res)
            # output, state = self.core(xt, fc_feats, state, att_res)
            logprobs = F.log_softmax(self.logit(output), dim=1)

            # sample the next word
            if t > 0:
                if sample_method == 'greedy':
                    sample_logprobs, sample_n = torch.max(logprobs.data, 1)
                    sample_n = sample_n.view(-1).long()
                else:
                    sample_logprobs, sample_n = sample_logprobs, it
                if decoding_constraint and t > 1:
                    tmp = logprobs.new_zeros(logprobs.size())
                    tmp.scatter_(1, seq[:, t - 2].data.unsqueeze(1), float('-inf'))
                    logprobs = logprobs + tmp

                seq[:, t - 1] = sample_n
                seqLogprobs[:, t - 1] = sample_logprobs.view(-1)
                if t == 1:
                    # initialize trigrams
                    trigrams = [{} for _ in range(batch_size)]

        return seq, seqLogprobs


class AoA_Module(nn.Module):
    def __init__(self, opt):
        super(AoA_Module, self).__init__()
        self.opt = opt
        self.att_hid_dim = opt.att_hid_dim

        # self.h2att = nn.Linear(self.rnn_size, self.att_hid_dim)
        # self.alpha_net = nn.Linear(self.att_hid_dim, 1)
        self.ctx2att = nn.Linear(opt.att_feat_size, self.att_hid_dim)

        self.q_bias = nn.Parameter(torch.zeros(self.att_hid_dim))
        self.x_bias = nn.Parameter(torch.zeros(self.att_hid_dim))

        self.query = nn.Linear(opt.rnn_size, self.att_hid_dim)
        self.key = nn.Linear(opt.att_feat_size, self.att_hid_dim)

        self.linear_q = nn.Linear(opt.rnn_size, self.att_hid_dim)
        self.linear_v = nn.Linear(opt.att_feat_size, self.att_hid_dim)

        self.information_enhancer = nn.Linear(self.att_hid_dim, self.att_hid_dim)
        self.information_enhancer_bias = nn.Parameter(torch.zeros(self.att_hid_dim))

        self.gate = nn.Linear(self.att_hid_dim, self.att_hid_dim)
        self.gate_bias = nn.Parameter(torch.zeros(self.att_hid_dim))

        self.attention = nn.Linear(self.att_hid_dim, 1)

    def forward(self, h, att_feats, att_masks=None):
        # The p_att_feats here is already projected
        # p_att_feats_ = p_att_feats.view(-1, self.num_heads, self.att_hid_dim)
        # p_att_feats_ = p_att_feats.permute(0, 2, 1)

        # query
        q = self.query(h)
        # q_ = self.linear_q(h)
        # q = q_.view(q_.size(0), -1, self.att_hid_dim).permute(0, 2, 1)

        # key
        k = self.key(att_feats)
        # k_ = self.linear_v(att_feats)
        # k = k_.view(k_.size(0), -1, self.att_hid_dim).permute(0, 2, 1)

        # value
        v = att_feats

        # Add-hoc components
        q = q.unsqueeze(1) + self.q_bias
        k = k + self.x_bias

        # # Additive Attention
        # q = q.unsqueeze(1) # [B, 1, D]
        # content_based_att = torch.tanh(q + k)

        # # Dot-product Attention
        # q = q.unsqueeze(1) # [B, 1, D]
        # content_based_att = torch.bmm(q, k.transpose(1, 2)) / self.att_hid_dim**0.5

        # Scaled Dot-product Attention
        q = q.unsqueeze(1)  # [B, 1, D]
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.att_hid_dim ** 0.5

        if att_masks is not None:
            attn = attn.masked_fill(att_masks.unsqueeze(1) == 0, -1e9)

        # content_based_att = self.attention(content_based_att)
        # content_based_att = content_based_att.squeeze(2)

        # if att_masks is not None:
        #     content_based_att = content_based_att.masked_fill(att_masks == 0, -1e9)

        # self.alpha = F.softmax(content_based_att, dim=1)
        self.alpha = F.softmax(attn, dim=2)

        # information_vector
        information_vector = torch.bmm(self.alpha, v)
        information_vector = information_vector.squeeze(1)

        # information_enhancer
        enhanced_information = torch.tanh(
            self.information_enhancer(information_vector) + self.information_enhancer_bias)

        # gating
        gate = torch.sigmoid(self.gate(enhanced_information) + self.gate_bias)

        att_res = gate * enhanced_information

        return att_res


class AoACore(nn.Module):
    def __init__(self, opt):
        super(AoACore, self).__init__()
        self.opt = opt
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.att_feat_size, opt.rnn_size)  # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size + opt.att_feat_size, opt.rnn_size)  # h^1_t, \hat v
        self.logit = nn.Linear(opt.rnn_size, opt.vocab_size + 1)

    def forward(self, xt, fc_feats, state, att_res):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        lang_lstm_input = torch.cat([att_res, h_att], 1)
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        # logprobs = F.log_softmax(self.logit(output), dim=1)

        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state