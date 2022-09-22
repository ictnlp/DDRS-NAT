# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import pdb
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from torch import Tensor
from nltk.translate.bleu_score import sentence_bleu
from collections import Counter

@register_criterion("nat_loss")
class LabelSmoothedDualImitationCriterion(FairseqCriterion):
    def __init__(self, task, label_smoothing):
        super().__init__(task)
        self.label_smoothing = label_smoothing
        self.args = self.task.args

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--label-smoothing",
            default=0.0,
            type=float,
            metavar="D",
            help="epsilon for label smoothing, 0 means no label smoothing",
        )
        parser.add_argument(
            '--ctc-ratio',
            type=float,
            default=0)
        parser.add_argument(
            '--tune',
            action="store_true")
        parser.add_argument(
            '--use-ngram',
            action="store_true")

    def _compute_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = (
                    nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def forward(self, model, sample, step = 0, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        losses, nll_loss = [], []

        for obj in outputs:
            if outputs[obj].get("loss", None) is None:
                _losses = self._compute_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", 0.0),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )
            else:
                _losses = self._custom_loss(
                    outputs[obj].get("loss"),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )

            losses += [_losses]
            if outputs[obj].get("nll_loss", False):
                nll_loss += [_losses.get("nll_loss", 0.0)]

        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 else loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))

        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )

        for key in logging_outputs[0]:
            if key[-5:] == "-loss":
                val = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key[:-5],
                    val / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("ctc_loss")
class CTCCriterion(LabelSmoothedDualImitationCriterion):
    def forward(self, model, sample, step = 0, reduce=True):
        #Compute the CTC loss for the given sample.
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens = sample["target"]
        tgt_lengths = torch.sum(sample["target"].ne(1), dim = -1)
        src_length = src_tokens.size(1)

        #construct decoder input
        if src_length > 200:
            max_length = 1 * src_length
        else:
            max_length = int(self.args.ctc_ratio * src_length)
        device = src_tokens.get_device()
        ctc_input_lengths = torch.LongTensor(nsentences).cuda(device)
        ctc_input_lengths[:] = max_length
        prev_output_tokens = torch.LongTensor(nsentences, max_length).cuda(device)
        prev_output_tokens[:] = 3
        prev_output_tokens[:,0] = 0
        prev_output_tokens[:,-1] = 2

        #calculate ctc loss
        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        log_probs = F.log_softmax(outputs['word_ins']['out'], dim=-1)
        log_probs = log_probs.transpose(0,1).float()
        loss = F.ctc_loss(log_probs, outputs['word_ins']['tgt'], 
            ctc_input_lengths, tgt_lengths, blank = 4, 
            reduction = 'mean', zero_infinity = True)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        return loss, sample_size, logging_output

@register_criterion("ddrs_loss")
class DDRSCriterion(LabelSmoothedDualImitationCriterion):

    def forward(self, model, sample, step = 0, reduce=True):
        #Compute the ctc loss with multiple references
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens = sample["target"]
        tgt_lengths = torch.sum(sample["target"].ne(1), dim = -1)
        src_length = src_tokens.size(1)

        #construct decoder input
        if src_length > 200:
            max_length = 1 * src_length
        else:
            max_length = int(self.args.ctc_ratio * src_length)
        device = src_tokens.get_device()
        ctc_input_lengths = torch.LongTensor(nsentences)
        ctc_input_lengths = ctc_input_lengths.cuda(device)
        ctc_input_lengths[:] = max_length
        prev_output_tokens = torch.LongTensor(nsentences, max_length)
        prev_output_tokens = prev_output_tokens.cuda(device)
        prev_output_tokens[:] = 3
        prev_output_tokens[:,0] = 0
        prev_output_tokens[:,-1] = 2

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        log_probs = F.log_softmax(outputs['word_ins']['out'], dim=-1)
        log_probs = log_probs.transpose(0,1)
        log_probs = log_probs.float()

        #if validation return ctc loss
        if not model.training:
            ctc_loss = F.ctc_loss(log_probs, outputs['word_ins']['tgt'],
                ctc_input_lengths, tgt_lengths, blank = 4, 
                reduction = 'mean', zero_infinity = True)
            sample_size = 1
            logging_output = {
                "loss": ctc_loss.data,
                "nll_loss": ctc_loss.data,
                "ntokens": ntokens,
                "nsentences": nsentences,
                "sample_size": sample_size,
            }
            return ctc_loss, sample_size, logging_output

        #extract references from target tokens
        div_tgt_tokens, div_tgt_lengths = self.extract_references(
            tgt_tokens, tgt_lengths, self.args.num_references, device, max_length)
        if not self.args.tune:
            loss = self.curr_loss(log_probs, div_tgt_tokens, div_tgt_lengths, 
                ctc_input_lengths, step)
        elsei:
            if self.args.use_ngram:
                loss = self.gram_loss(log_probs, div_tgt_tokens, ctc_input_lengths, model.tgt_dict)
            else:
                loss = self.rl_loss(log_probs, div_tgt_tokens, ctc_input_lengths, model.tgt_dict)

        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def gram_loss(self, log_probs, div_tgt_tokens, ctc_input_lengths, tgt_dict):

        log_probs = log_probs.transpose(0,1)
        batch_size, length_ctc, vocab_size = log_probs.size()
        probs = torch.exp(log_probs)
        num_refs = len(div_tgt_tokens)
        for i in range(num_refs):
            div_tgt_tokens[i] = div_tgt_tokens[i].tolist()
        probs_blank = probs[:,:,4]
        length = probs[:,1:,:] * (1 - probs[:,:-1,:])
        length[:,:,4] = 0
        expected_length = torch.sum(length).div(batch_size)

        logprobs_blank = log_probs[:,:,4]
        cumsum_blank = torch.cumsum(logprobs_blank, dim = -1)
        cumsum_blank_A = cumsum_blank.view(batch_size, 1, length_ctc).expand(-1, length_ctc, -1)
        cumsum_blank_B = cumsum_blank.view(batch_size, length_ctc, 1).expand(-1, -1, length_ctc)
        cumsum_blank_sub = cumsum_blank_A - cumsum_blank_B
        cumsum_blank_sub = torch.cat((torch.zeros(batch_size, length_ctc,1).cuda(cumsum_blank_sub.get_device()), cumsum_blank_sub[:,:,:-1]), dim = -1)
        tri_mask = torch.tril(utils.fill_with_neg_inf(torch.zeros([batch_size, length_ctc, length_ctc]).cuda(cumsum_blank_sub.get_device())), 0)
        cumsum_blank_sub = cumsum_blank_sub + tri_mask
        blank_matrix = torch.exp(cumsum_blank_sub)

        gram_1 = []
        gram_2 = []
        gram_count = []
        rep_gram_pos = []
        num_grams = 0
        for i in range(batch_size):
            two_grams = Counter()
            gram_1.append([])
            gram_2.append([])
            gram_count.append([])
            for j in range(num_refs):
                curr_tgt = div_tgt_tokens[j][i]
                for k in range(len(curr_tgt) - 1):
                    if curr_tgt[k+1] != 1:
                        two_grams[(curr_tgt[k], curr_tgt[k+1])] += 1/num_refs
            for two_gram in two_grams:
                gram_1[-1].append(two_gram[0])
                gram_2[-1].append(two_gram[1])
                gram_count[-1].append(two_grams[two_gram])
            num_grams = max(len(gram_count[-1]), num_grams)
        
        for i in range(batch_size):
            while len(gram_count[i]) < num_grams:
                gram_1[i].append(1)
                gram_2[i].append(1)
                gram_count[i].append(0)
        gram_1 = torch.LongTensor(gram_1).cuda(blank_matrix.get_device())
        gram_2 = torch.LongTensor(gram_2).cuda(blank_matrix.get_device())
        gram_count = torch.Tensor(gram_count).cuda(blank_matrix.get_device()).view(batch_size, num_grams,1)
        gram_1_probs = torch.gather(probs, -1, gram_1.view(batch_size,1,num_grams).expand(batch_size,length_ctc,num_grams)).transpose(1,2).view(batch_size, num_grams, length_ctc, 1)
        gram_2_probs = torch.gather(probs, -1, gram_2.view(batch_size,1,num_grams).expand(batch_size,length_ctc,num_grams)).transpose(1,2).view(batch_size, num_grams, 1, length_ctc)
        probs_matrix = torch.matmul(gram_1_probs, gram_2_probs)
        bag_grams = blank_matrix.view(batch_size, 1, length_ctc, length_ctc) * probs_matrix
        bag_grams = torch.sum(bag_grams.view(batch_size, num_grams, -1), dim = -1).view(batch_size, num_grams,1)
        match_gram = torch.min(torch.cat([bag_grams,gram_count],dim = -1), dim = -1)[0]
        match_gram = torch.sum(match_gram).div(batch_size)
        loss = (- 2 * match_gram).div(torch.sum(gram_count).div(batch_size) + expected_length - 1)
        return loss

    def curr_loss(self, log_probs, div_tgt_tokens, div_tgt_lengths, ctc_input_lengths, step):
        num_references = len(div_tgt_tokens)
        ctc_losses = []
        for i in range(num_references):
            ctc_losses.append(F.ctc_loss(log_probs, div_tgt_tokens[i] , ctc_input_lengths,
             div_tgt_lengths[i], blank = 4, reduction = 'none', zero_infinity = True))
        ctc_losses = torch.stack(ctc_losses, dim = 0)

        #leave some steps for checkpoint averaging
        time = step / (self.args.max_update - self.save_interval_updates * 4)
        curr_lambda = 2/3
        if time < curr_lambda:
            t_1 = time / curr_lambda
            ctc_sum_loss = torch.sum(ctc_losses).div(torch.sum(div_tgt_lengths))
            ctc_lse_loss = - torch.sum(torch.logsumexp(-ctc_losses, dim = 0))
            ctc_lse_loss = ctc_lse_loss.div(torch.sum(div_tgt_lengths) / num_references)
            loss = t_1 * ctc_lse_loss + (1 - t_1) * ctc_sum_loss
        elif time < 1:
            t_2 = (time - curr_lambda) / (1 - curr_lambda)
            ctc_lse_loss = - torch.sum(torch.logsumexp(-ctc_losses, dim = 0))
            ctc_lse_loss = ctc_lse_loss.div(torch.sum(div_tgt_lengths) / num_references)
            ctc_min_loss, min_idx = torch.min(ctc_losses, dim = 0)
            min_tgt_lengths = torch.gather(div_tgt_lengths, 0, min_idx.view(1, -1))
            ctc_min_loss = torch.sum(ctc_min_loss).div(torch.sum(min_tgt_lengths))
            loss = t_2 * ctc_min_loss + (1 - t_2) * ctc_lse_loss
        else:
            ctc_min_loss, min_idx = torch.min(ctc_losses, dim = 0)
            min_tgt_lengths = torch.gather(div_tgt_lengths, 0, min_idx.view(1, -1))
            ctc_min_loss = torch.sum(ctc_min_loss).div(torch.sum(min_tgt_lengths))
            loss = ctc_min_loss

        return loss

    def rl_loss(self, log_probs, div_tgt_tokens, ctc_input_lengths, tgt_dict):
        log_probs = log_probs.transpose(0, 1)
        nsentences, output_length, vocab_size = log_probs.size()
        probs = torch.exp(log_probs).reshape(nsentences * output_length, vocab_size)
        targets = [target.tolist() for target in div_tgt_tokens]
        num_references = len(targets)
        for i in range(num_references):
            for j in range(nsentences):
                targets[i][j] = [idx for idx in targets[i][j]
                if idx != 0 and idx != 1 and idx != 2]

        #sample 20 sentences, calculate rewards
        sample_times = 20
        all_sample_rewards = []
        all_sample_sentences = []
        for i in range(sample_times):
            sample_outs = torch.multinomial(probs, 1).view(nsentences, output_length)
            sample_outs = sample_outs.tolist()
            sample_sentences = []
            for j in range(nsentences):
                #example sample_out: A A blank A B B C, sample_sentence: A A B C
                sample_out = [idx for idx in sample_outs[j]
                    if idx != 0 and idx != 1 and idx != 2 and idx != 4]
                if len(sample_out) != 0:
                    sample_sentence = [sample_out[0]]
                else:
                    sample_sentence = []
                for k in range(len(sample_out) - 1):
                    if sample_out[k + 1] != sample_out[k]:
                        sample_sentence.append(sample_out[k + 1])
                sample_sentences.append(sample_sentence)
            all_sample_sentences.append(sample_sentences)
            all_sample_rewards.append(self.compute_reward(sample_sentences, targets, tgt_dict))

        device = probs.get_device()
        all_sample_rewards = torch.stack(all_sample_rewards, dim = 0).cuda(device)
        mean_reward = torch.mean(all_sample_rewards, dim = 0).view(1, nsentences)

        #trick: select the best sample to update model
        max_reward, max_idx = torch.max(all_sample_rewards, dim = 0)
        max_sample_index = []
        for i in range(nsentences):
            max_sample_index.append([word for word in all_sample_sentences[max_idx[i]][i]])
        max_sample_index, sample_lengths = self.padsample(max_sample_index, device)

        #scale the reward
        reward_variance = torch.mean((all_sample_rewards - mean_reward)**2,dim = 0)
        reward_mu = torch.sqrt(reward_variance) + 1e-4
        reward = (max_reward - mean_reward.view(-1)) / reward_mu

        ctc_logprob = F.ctc_loss(log_probs.transpose(0,1), max_sample_index, ctc_input_lengths,
             sample_lengths, blank = 4, reduction = 'none', zero_infinity = True)
        loss = sum(ctc_logprob * reward) / sum(sample_lengths)

        return loss

    def compute_reward(self, sample_sentences, targets, tgt_dict):

        rewards = []
        num_references = len(targets)
        nsentences = len(sample_sentences)

        for i in range(nsentences):
            #calculte reward according to multiple references
            ref_rewards = []
            sample_sentence = ""
            for word in sample_sentences[i]:
                sample_sentence += tgt_dict[word] + ' '
            sample_sentence = sample_sentence[:-1].split(' ')
            for j in range(num_references):
                target_sentence = ""
                for word in targets[j][i]:
                    target_sentence += tgt_dict[word] + ' '
                target_sentence = target_sentence[:-1].split(' ')
                reward = sentence_bleu([target_sentence], sample_sentence)
                ref_rewards.append(reward)
            #max reward RL
            reward = max(ref_rewards)
            rewards.append(reward)
        rewards = torch.Tensor(rewards)
        return rewards

    def padsample(self, sample_index, device):

        len_samples = [len(sample) + 2 for sample in sample_index]
        max_length = max(len_samples)
        for i in range(len(sample_index)):
            sample_index[i].insert(0, 0)
            sample_index[i].append(2)
            for j in range(max_length - len_samples[i]):
                sample_index[i].append(1)
        sample_index = torch.LongTensor(sample_index).cuda(device)
        len_samples = torch.LongTensor(len_samples).cuda(device)
        return sample_index, len_samples

    def extract_references(self, tgt_tokens, tgt_lengths, num_references, device, max_length):
        tgt_tokens = tgt_tokens.tolist()
        tgt_lengths = tgt_lengths.tolist()
        div_tgt_tokens = []
        nsentences = len(tgt_tokens)

        #special tokens, 0:bos, 1:pad, 2:eos, 3:unk, 4:blank, 5:divide
        bos_token = 0
        pad_token = 1
        eos_token = 2
        divide_token = 5

        #target format: ref_1 divide_token ref_2 divide_token ... ref_n
        #extract references from target tokens
        for i in range(num_references):
            div_tgt_tokens.append([])
        for i in range(nsentences):
            ref_id = 0
            div_tgt_tokens[ref_id].append([])
            for j in range(tgt_lengths[i]):
                if tgt_tokens[i][j] != divide_token:
                    if len(div_tgt_tokens[ref_id][i]) < max_length - 1:
                        div_tgt_tokens[ref_id][i].append(tgt_tokens[i][j])
                else:
                    div_tgt_tokens[ref_id][i].append(eos_token)
                    ref_id += 1
                    div_tgt_tokens[ref_id].append([])
                    div_tgt_tokens[ref_id][i].append(bos_token)
        max_lens = []
        for i in range(num_references):
            max_lens.append(max([len(sentence) for sentence in div_tgt_tokens[i]]))

        #add padding
        div_tgt_lengths = []
        for i in range(num_references):
            div_tgt_lengths.append([])
            for j in range(nsentences):
                l = len(div_tgt_tokens[i][j])
                div_tgt_lengths[i].append(l)
                for k in range(max_lens[i] - l):
                    div_tgt_tokens[i][j].append(pad_token)

        div_tgt_tokens = [torch.LongTensor(div_tgt_token).cuda(device) 
            for div_tgt_token in div_tgt_tokens]
        div_tgt_lengths = torch.LongTensor(div_tgt_lengths).cuda(device)
        return div_tgt_tokens, div_tgt_lengths
