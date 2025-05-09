# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 09:47:54 2020

@author: HQ Xie
utils.py
"""
import os 
import math
import torch
import time
import torch.nn as nn
import numpy as np
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
from models.mutual_info import sample_batch, mutual_information

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1 # 1-gram weights
        self.w2 = w2 # 2-grams weights
        self.w3 = w3 # 3-grams weights
        self.w4 = w4 # 4-grams weights
    
    def compute_blue_score(self, real, predicted):
        score = []
        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            if not sent1 or not sent2:
                score.append(0.0)
                continue
            score.append(sentence_bleu([sent1], sent2, 
                          weights=(self.w1, self.w2, self.w3, self.w4)))
        return score
            

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target, criterion_external):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2 if self.size > 2 else 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence) 
        true_dist[:, self.padding_idx] = 0 
        mask = torch.nonzero(target.data == self.padding_idx, as_tuple=False) # as_tuple=False for older torch
        if mask.dim() > 0 and mask.size(0) > 0: # Check if mask is not empty
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return criterion_external(x, true_dist)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        if self.warmup == 0: # Handle warmup=0 case to avoid division by zero or large LR
             if step == 0 : return 0.0 # or some initial small LR
             return self.factor * (self.model_size ** (-0.5) * step ** (-0.5))

        lr = self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5) if step > 0 else float('inf'), step * self.warmup ** (-1.5)))
        return lr
    

            
class SeqtoText:
    def __init__(self, vocb_dictionary, end_idx):
        self.reverse_word_map = dict(zip(vocb_dictionary.values(), vocb_dictionary.keys()))
        self.end_idx = end_idx 
        
    def sequence_to_text(self, list_of_indices):
        words = []
        for idx_val in list_of_indices: 
            if isinstance(idx_val, torch.Tensor): # Handle tensor elements
                idx_val = idx_val.item()
            if idx_val == self.end_idx:
                break

            word = self.reverse_word_map.get(idx_val, '<UNK>') 
            words.append(word)
        return ' '.join(words)


class Channels():
    def AWGN(self, Tx_sig, n_var_val): 

        noise = torch.normal(0, n_var_val, size=Tx_sig.shape).to(Tx_sig.device)
        Rx_sig = Tx_sig + noise
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_var_val):
        shape = Tx_sig.shape
        H_real = torch.normal(0, math.sqrt(1/2), size=(shape[0], 1, 1), device=Tx_sig.device).expand_as(Tx_sig[...,0:1])
        H_imag = torch.normal(0, math.sqrt(1/2), size=(shape[0], 1, 1), device=Tx_sig.device).expand_as(Tx_sig[...,1:2] if Tx_sig.shape[-1]==2 else Tx_sig[...,0:1]) # Assuming Tx_sig last dim is 2 (real, imag) or 1
        
        if Tx_sig.shape[-1] != 2:
            H_amp = torch.sqrt(H_real**2 + H_imag**2)
            Tx_sig_faded = Tx_sig * H_amp 
        else:
            H_matrix_real = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
            H_matrix_imag = torch.normal(0, math.sqrt(1/2), size=[1]).to(device)
            H_matrix = torch.Tensor([[H_matrix_real, -H_matrix_imag], [H_matrix_imag, H_matrix_real]]).to(device)
            
            Tx_sig_view = Tx_sig.reshape(shape[0] * shape[1], 2)
            Tx_sig_faded_view = torch.matmul(Tx_sig_view, H_matrix)
            Tx_sig_faded = Tx_sig_faded_view.reshape(shape)

        Rx_sig = self.AWGN(Tx_sig_faded, n_var_val)
        # Channel estimation - Perfect CSI assumed
        if Tx_sig.shape[-1] == 2:
            Rx_sig_view = Rx_sig.reshape(shape[0] * shape[1], 2)
            # Check if H_matrix is invertible
            if torch.det(H_matrix) == 0:
                # print("Warning: H_matrix is singular, cannot invert. Skipping equalization.")
                Rx_sig = Rx_sig # Or handle differently
            else:
                Rx_sig_compensated_view = torch.matmul(Rx_sig_view, torch.inverse(H_matrix))
                Rx_sig = Rx_sig_compensated_view.reshape(shape)

        return Rx_sig

    def Rician(self, Tx_sig, n_var_val, K=1): # n_var_val
        shape = Tx_sig.shape
        # Line of Sight (LoS) component
        los_real = math.sqrt(K / (K + 1)) 
        los_imag = 0
        # Scattered component (Rayleigh)
        scatter_std = math.sqrt(1 / (2 * (K + 1))) 
        H_real_val = los_real + torch.normal(0, scatter_std, size=[1]).to(Tx_sig.device)
        H_imag_val = los_imag + torch.normal(0, scatter_std, size=[1]).to(Tx_sig.device)
        
        if Tx_sig.shape[-1] != 2:
            # print("Warning: Rician channel expects last dimension of Tx_sig to be 2 (real, imag).")
            H_amp = torch.sqrt(H_real_val**2 + H_imag_val**2)
            Tx_sig_faded = Tx_sig * H_amp
        else:
            H_matrix = torch.Tensor([[H_real_val.item(), -H_imag_val.item()], 
                                     [H_imag_val.item(), H_real_val.item()]]).to(Tx_sig.device)
            Tx_sig_view = Tx_sig.reshape(shape[0] * shape[1], 2)
            Tx_sig_faded_view = torch.matmul(Tx_sig_view, H_matrix)
            Tx_sig_faded = Tx_sig_faded_view.reshape(shape)

        Rx_sig = self.AWGN(Tx_sig_faded, n_var_val)
        
        # Channel estimation (Perfect CSI)
        if Tx_sig.shape[-1] == 2:
            Rx_sig_view = Rx_sig.reshape(shape[0] * shape[1], 2)
            if torch.det(H_matrix) == 0:
                # print("Warning: H_matrix for Rician is singular. Skipping equalization.")
                Rx_sig = Rx_sig
            else:
                Rx_sig_compensated_view = torch.matmul(Rx_sig_view, torch.inverse(H_matrix))
                Rx_sig = Rx_sig_compensated_view.reshape(shape)
        # elif Tx_sig.shape[-1] != 2 and H_amp != 0: Rx_sig = Rx_sig / H_amp
        return Rx_sig


def initNetParams(model):
    for m in model.modules(): # Iterate over all modules, not just direct parameters
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    return model
         
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask_np = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask_np)

    
def create_masks(src, trg, current_pad_idx): 
    src_mask = (src == current_pad_idx).unsqueeze(1).unsqueeze(2) # Shape: (batch_size, 1, 1, src_len) for MHA
    trg_combined_mask = None
    if trg is not None:
        trg_pad_mask = (trg == current_pad_idx).unsqueeze(1).unsqueeze(2) # Shape: (batch_size, 1, 1, trg_len)
        look_ahead_mask = subsequent_mask(trg.size(-1)).to(trg.device) 
        
        trg_combined_mask = torch.max(trg_pad_mask, look_ahead_mask) 
    
    src_mask_original = (src == current_pad_idx).unsqueeze(-2).type(torch.FloatTensor).to(device) #[batch, 1, seq_len]
    trg_combined_mask_original = None
    if trg is not None:
        trg_mask_original = (trg == current_pad_idx).unsqueeze(-2).type(torch.FloatTensor).to(device) #[batch, 1, seq_len]
        look_ahead_mask_original = subsequent_mask(trg.size(-1)).type_as(trg_mask_original.data).to(device)
        trg_combined_mask_original = torch.max(trg_mask_original, look_ahead_mask_original)

    return src_mask_original, trg_combined_mask_original


def loss_function(pred_logits, target_indices, current_pad_idx, criterion_ext): # pred_logits, target_indices, current_pad_idx, criterion_ext
    # pred_logits: (batch_size * seq_len, vocab_size)
    # target_indices: (batch_size * seq_len)
    
    loss_all_tokens = criterion_ext(pred_logits, target_indices) # (batch_size * seq_len)
    
    mask = (target_indices != current_pad_idx).type_as(loss_all_tokens.data)
    
    loss_masked = loss_all_tokens * mask
    
    non_pad_elements = mask.sum()
    if non_pad_elements > 0:
        return loss_masked.sum() / non_pad_elements
    return torch.tensor(0.0).to(pred_logits.device)


def PowerNormalize(x):
    x_square = torch.mul(x, x) 
    power = torch.mean(x_square).sqrt()

    if power > 1.0:
        x = torch.div(x, power)
    return x


def SNR_to_noise(snr_db_val): # snr_db_val
    snr_linear = 10 ** (snr_db_val / 10.0)
    noise_std_val = 1.0 / np.sqrt(2 * snr_linear)
    return noise_std_val

def train_step(model, src, trg, n_var, current_pad_idx, opt, current_criterion, channel, 
               lambda_mi_coeff, mi_net=None, return_mi_contribution=False):
    model.train()

    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    channels_obj = Channels()
    opt.zero_grad()
    
    src_mask, look_ahead_mask = create_masks(src, trg_inp, current_pad_idx)
    
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels_obj.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels_obj.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels_obj.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred_logits = model.dense(dec_output)
    
    pred_logits_flat = pred_logits.contiguous().view(-1, pred_logits.size(-1))
    trg_real_flat = trg_real.contiguous().view(-1)
    
    loss_ce = loss_function(pred_logits_flat, trg_real_flat, 
                            current_pad_idx, current_criterion)

    total_loss = loss_ce
    mi_loss_value_contrib = 0.0

    if mi_net is not None and lambda_mi_coeff > 0:
        mi_net.eval()
        
        joint, marginal = sample_batch(Tx_sig, Rx_sig) 

        mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
        loss_mine_term = -mi_lb
        
        mi_loss_value_contrib = lambda_mi_coeff * loss_mine_term.item()
        
        total_loss = total_loss + lambda_mi_coeff * loss_mine_term
    
    total_loss.backward()
    opt.step()

    if return_mi_contribution:
        return total_loss.item(), mi_loss_value_contrib
    else:
        return total_loss.item()

def train_mi(model_deepsc, model_mine, src, n_var, current_pad_idx, optimizer_mine, channel): 
    model_mine.train()
    model_deepsc.eval()

    optimizer_mine.zero_grad()
    channels_obj = Channels()
    
    src_mask, _ = create_masks(src, None, current_pad_idx)
    
    with torch.no_grad():
        enc_output = model_deepsc.encoder(src, src_mask)
        channel_enc_output = model_deepsc.channel_encoder(enc_output)
        Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels_obj.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels_obj.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels_obj.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    joint, marginal = sample_batch(Tx_sig.detach(), Rx_sig.detach())
    
    mi_lb, _, _ = mutual_information(joint, marginal, model_mine)
    loss_for_mine_training = -mi_lb

    loss_for_mine_training.backward()
    optimizer_mine.step()

    return mi_lb.item()

def val_step(model, src, trg, n_var, current_pad_idx, current_criterion, channel):
    model.eval()
    channels_obj = Channels()
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    src_mask, look_ahead_mask = create_masks(src, trg_inp, current_pad_idx)
    
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if channel == 'AWGN':
        Rx_sig = channels_obj.AWGN(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician for validation")


    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred_logits = model.dense(dec_output)

    pred_logits_flat = pred_logits.contiguous().view(-1, pred_logits.size(-1))
    trg_real_flat = trg_real.contiguous().view(-1)
    
    loss = loss_function(pred_logits_flat, trg_real_flat, 
                         current_pad_idx, current_criterion)
    
    return loss.item()
    
def greedy_decode(model, src, n_var, max_len_decode, current_pad_idx, current_start_idx, channel): # max_len_decode
    model.eval()
    channels_obj = Channels()
    
    src_mask, _ = create_masks(src, None, current_pad_idx)

    with torch.no_grad():
        enc_output = model.encoder(src, src_mask)
        channel_enc_output = model.channel_encoder(enc_output)
        Tx_sig = PowerNormalize(channel_enc_output)

        if channel == 'AWGN':
            Rx_sig = channels_obj.AWGN(Tx_sig, n_var)
        else:
            raise ValueError("Please choose from AWGN, Rayleigh, and Rician for greedy_decode")
            
        memory_from_channel_decoder = model.channel_decoder(Rx_sig)
        
        outputs = torch.ones(src.size(0), 1, dtype=torch.long, device=src.device).fill_(current_start_idx)

        for _ in range(max_len_decode - 1):
            _, look_ahead_dec_mask = create_masks(None, outputs, current_pad_idx)
            
            dec_output = model.decoder(outputs, memory_from_channel_decoder, look_ahead_dec_mask, src_mask)
            pred_logits_step = model.dense(dec_output)

            next_word_logits = pred_logits_step[: ,-1:, :]
            
            _, next_word_indices = torch.max(next_word_logits, dim = -1)
            
            outputs = torch.cat([outputs, next_word_indices], dim=1)

    return outputs