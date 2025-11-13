import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision.models import densenet
from torchvision.models import resnet

densenet_versions = {
    'densenet121': densenet.densenet121,
    'densenet161': densenet.densenet161,
    'densenet169': densenet.densenet169,
    'densenet201': densenet.densenet201,
}

resnet_versions = {
    'resnet50': resnet.resnet50,
    'resnet101': resnet.resnet101,
    'resnet152': resnet.resnet152,
}


def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed = pack_padded_sequence(att_feats, list(att_masks.data.long().sum(1)), batch_first=True)
        return pad_packed_sequence(module(packed[0]), batch_first=True)[0]
    else:
        return module(att_feats)


class AttModel(nn.Module):
    def __init__(self, opt):
        super(AttModel, self).__init__()
        self.opt = opt
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.use_bn = getattr(opt, 'use_bn', 0)
        self.ss_prob = 0.0
        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                   nn.ReLU(),
                                   nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                      nn.ReLU(),
                                      nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.rnn_size),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm)) +
                ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn == 2 else ())))
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        else:
            self.logit = [[nn.Linear(self.rnn_size, self.rnn_size), nn.ReLU(), nn.Dropout(0.5)] for _ in
                          range(self.logit_layers - 1)]
            self.logit = nn.Sequential(
                *(reduce(lambda x, y: x + y, self.logit) + [nn.Linear(self.rnn_size, self.vocab_size + 1)]))

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        p_att_feats = self.ctx2att(att_feats)
        return fc_feats, att_feats, p_att_feats, att_masks

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        bsz = fc_feats.size(0)
        state = self.init_hidden(bsz)
        outputs = []
        fc_feats, att_feats, p_att_feats, att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        for i in range(seq.size(1)):
            if self.training and i >= 1 and self.ss_prob > 0.0:
                sample_prob = fc_feats.new(bsz).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[-1].data)
                    it.index_copy_(0, sample_ind,
                                   torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()
            if i >= 1 and seq[:, i].sum() == 0:
                continue
            output, state = self.get_logprobs_state(it, fc_feats, att_feats, p_att_feats, att_masks, state)
            outputs.append(output)
        return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state):
        xt = self.embed(it)
        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        logprobs = F.log_softmax(self.logit(output), dim=1)
        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt=None):
        beam_size = opt.get('beam_size', 10)
        bsz = fc_feats.size(0)
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case needs to be handled'
        seq = torch.LongTensor(self.seq_length, bsz).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, bsz)
        self.done_beams = [[] for _ in range(bsz)]
        for k in range(bsz):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = p_fc_feats[k:k + 1].expand(beam_size, p_fc_feats.size(1))
            tmp_att_feats = p_att_feats[k:k + 1].expand(*((beam_size,) + p_att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = pp_att_feats[k:k + 1].expand(*((beam_size,) + pp_att_feats.size()[1:])).contiguous()
            tmp_att_masks = p_att_masks[k:k + 1].expand(
                beam_size, p_att_masks.size(1)) if att_masks is not None else None
            it = fc_feats.new_zeros(beam_size, dtype=torch.long)
            logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks,
                                                      state)
            self.done_beams[k] = self.beam_step(logprobs, beam_size, it, state)
            for i in range(2, self.seq_length + 1):
                if len(self.done_beams[k]) == beam_size:
                    break
                logprobs, state = self.get_logprobs_state(self.done_beams[k][0][1], tmp_fc_feats, tmp_att_feats,
                                                          tmp_p_att_feats, tmp_att_masks, self.done_beams[k][0][2])
                self.done_beams[k] = self.beam_step(logprobs, beam_size, self.done_beams[k][0][1], state)
            seq[:, k] = self.done_beams[k][0][1]
            seqLogprobs[:, k] = self.done_beams[k][0][0]
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, fc_feats, att_feats, att_masks=None, opt=None):
        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)
        bsz = fc_feats.size(0)
        state = self.init_hidden(bsz)
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)
        trigrams = []
        seq = fc_feats.new_zeros((bsz, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(bsz, self.seq_length)
        for t in range(self.seq_length + 1):
            if t == 0:
                it = fc_feats.new_zeros(bsz, dtype=torch.long)
            elif sample_method == 'greedy':
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data)
                else:
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it)
                it = it.view(-1).long()
            if decoding_constraint and it.item() > 0 and it.item() <= self.vocab_size:
                try:
                    event = self.ix_to_word[str(it.item())]
                except:
                    event = self.ix_to_word[it.item()]
            if block_trigrams and t >= 3:
                prev_two_batch = seq[:, t - 3:t - 1]
                for i in range(bsz):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        logprobs[i, it[i]] = -float('inf')
            if t < self.seq_length:
                seq[:, t] = it
                seqLogprobs[:, t] = sampleLogprobs.view(-1)
            if t >= 2:
                prev_two = (seq[t - 2], seq[t - 1])
                trigrams.append(prev_two)
            logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)
        return seq, seqLogprobs


class MyResnet(nn.Module):
    def __init__(self, cnn_model='resnet101', pretrained=False):
        super(MyResnet, self).__init__()
        if cnn_model not in resnet_versions:
            print("Warning: CNN model '{}' not recognized. "
                  "Defaulting to resnet101.".format(cnn_model))
            cnn_model = 'resnet101'
        self.cnn = resnet_versions[cnn_model](pretrained=pretrained)
        self.cnn_model = cnn_model

    def forward(self, img):
        if 'resnet' in self.cnn_model:
            x = self.cnn.conv1(img)
            x = self.cnn.bn1(x)
            x = self.cnn.relu(x)
            x = self.cnn.maxpool(x)
            x = self.cnn.layer1(x)
            x = self.cnn.layer2(x)
            x = self.cnn.layer3(x)
            x = self.cnn.layer4(x)
        x = x.view(x.size(0), x.size(1), -1).transpose(1, 2)
        return x


class MyDensenet(nn.Module):
    def __init__(self, cnn_model='densenet161', pretrained=True):
        super(MyDensenet, self).__init__()
        if cnn_model not in densenet_versions:
            print("Warning: CNN model '{}' not recognized. "
                  "Defaulting to densenet161.".format(cnn_model))
            cnn_model = 'densenet161'
        self.cnn = densenet_versions[cnn_model](pretrained=pretrained)
        self.cnn_model = cnn_model

    def forward(self, img):
        x = self.cnn.features(img)
        x = x.view(x.size(0), x.size(1), -1).transpose(1, 2)
        return x