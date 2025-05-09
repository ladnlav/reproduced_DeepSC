# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:59:14 2020

@author: HQ Xie
"""
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils_new import SNR_to_noise, initNetParams, train_step, val_step, train_mi 
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from models.mutual_info import Mine 
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', default='data/europarl_processed/vocab.json', type=str) 
parser.add_argument('--checkpoint-path', default='checkpoints/default_run', type=str) 
parser.add_argument('--channel', default='AWGN', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int) 
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=30, type=int, help='Default epochs for joint training or if phase-specific epochs not set.')
parser.add_argument('--lambda_mi', default=0.0009, type=float, help='Weight for the MI loss component')
parser.add_argument('--training_phase', type=int, default=0, 
                    help='0: Joint training (current), 1: Phase 1 (train MI net only), 2: Phase 2 (train DeepSC with frozen MI net)')
parser.add_argument('--epochs_phase1', type=int, default=20, help='Epochs for Phase 1 (MI Net training)')
parser.add_argument('--epochs_phase2', type=int, default=50, help='Epochs for Phase 2 (DeepSC training with frozen MI Net)')
parser.add_argument('--deepsc_checkpoint_load_path', type=str, default=None, help='Path to load DeepSC checkpoint')
parser.add_argument('--mi_net_checkpoint_load_path', type=str, default=None, help='Path to load a pre-trained MI net checkpoint')
parser.add_argument('--plot_curves', action='store_true', help='Plot training curves at the end of training.') # <--- ДОБАВЛЕНО


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pad_idx_global = None
criterion_global = None
optimizer_deepsc_global = None 
mi_optimizer_global = None   

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def validate(epoch, args, net, val_iterator):
    net.eval()
    pbar = tqdm(val_iterator, desc=f"Epoch {epoch+1} [Validation]")
    total_loss = 0
    num_batches = 0
    validation_noise_std = SNR_to_noise(10) 

    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            loss = val_step(net, sents, sents, validation_noise_std, pad_idx_global,
                             criterion_global, args.channel)
            total_loss += loss
            num_batches += 1
            pbar.set_postfix_str(f"Loss: {loss:.5f}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    print(f"Epoch {epoch+1} [Validation]: Average Loss: {avg_loss:.5f}")
    return avg_loss

def train_phase1_mi_net_only(epoch, args, deepsc_model, mi_network, train_iterator):
    global mi_optimizer_global 
    deepsc_model.eval() 
    mi_network.train()  
    
    pbar = tqdm(train_iterator, desc=f"Epoch {epoch+1} [Phase 1: Train MI Net]")
    total_mi_val = 0
    num_batches = 0

    for sents in pbar:
        sents = sents.to(device)
        noise_std_for_mi_training = np.random.uniform(SNR_to_noise(0), SNR_to_noise(25), size=(1))[0]
        mi_val = train_mi(deepsc_model, mi_network, sents, noise_std_for_mi_training, 
                          pad_idx_global, mi_optimizer_global, args.channel)
        total_mi_val += mi_val
        num_batches += 1
        pbar.set_postfix_str(f"Current MI_lb: {mi_val:.5f}")
    
    avg_mi_val = total_mi_val / num_batches if num_batches > 0 else float('-inf')
    print(f"Epoch {epoch+1} [Phase 1]: Average MI_lb evaluated by MI Net: {avg_mi_val:.5f}")
    return avg_mi_val

def train_phase2_deepsc_only(epoch, args, deepsc_model, frozen_mi_network, train_iterator):
    global optimizer_deepsc_global, criterion_global 
    deepsc_model.train() 
    if frozen_mi_network:
        frozen_mi_network.eval() 

    pbar = tqdm(train_iterator, desc=f"Epoch {epoch+1} [Phase 2: Train DeepSC]")
    epoch_total_loss = 0
    epoch_total_mi_contrib = 0
    num_batches = 0

    for sents in pbar:
        sents = sents.to(device)
        noise_std_for_deepsc_training = np.random.uniform(SNR_to_noise(0), SNR_to_noise(25), size=(1))[0]
        
        current_loss, mi_contribution = train_step(
            deepsc_model, sents, sents, noise_std_for_deepsc_training, pad_idx_global,
            optimizer_deepsc_global, criterion_global, args.channel, 
            args.lambda_mi, frozen_mi_network, return_mi_contribution=True
        )
        
        epoch_total_loss += current_loss
        if mi_contribution is not None:
             epoch_total_mi_contrib += mi_contribution
        num_batches +=1
        
        if frozen_mi_network and args.lambda_mi > 0 and mi_contribution is not None:
            pbar.set_postfix_str(f"Total Loss: {current_loss:.5f}, MI_contrib: {mi_contribution:.5f}")
        else:
            pbar.set_postfix_str(f"Total Loss: {current_loss:.5f}")
    
    avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else float('inf')
    avg_epoch_mi_contrib = epoch_total_mi_contrib / num_batches if num_batches > 0 and args.lambda_mi > 0 else 0.0
    
    print(f"Epoch {epoch+1} [Phase 2]: Average Total Loss: {avg_epoch_loss:.5f}")
    if frozen_mi_network and args.lambda_mi > 0:
        print(f"Epoch {epoch+1} [Phase 2]: Average MI Contribution to Loss: {avg_epoch_mi_contrib:.5f}")
    return avg_epoch_loss, avg_epoch_mi_contrib

def train_joint(epoch, args, net, mi_network, train_iterator):
    global optimizer_deepsc_global, mi_optimizer_global, criterion_global 
    net.train() 
    if mi_network and args.lambda_mi > 0: 
        mi_network.train() 

    pbar = tqdm(train_iterator, desc=f"Epoch {epoch+1} [Joint Training]")
    epoch_total_loss = 0
    epoch_total_mi_val_train_mi = 0
    epoch_total_mi_contrib_loss = 0
    num_batches = 0

    for sents in pbar:
        sents = sents.to(device)
        noise_std_for_training = np.random.uniform(SNR_to_noise(0), SNR_to_noise(25), size=(1))[0]
        
        current_mi_val = 0.0
        mi_contribution_to_loss_val = 0.0

        if mi_network and args.lambda_mi > 0: 
            current_mi_val = train_mi(net, mi_network, sents, noise_std_for_training, 
                                      pad_idx_global, mi_optimizer_global, args.channel)
            
        current_loss, mi_contribution_to_loss_val = train_step(
            net, sents, sents, noise_std_for_training, pad_idx_global,
            optimizer_deepsc_global, criterion_global, args.channel, 
            args.lambda_mi, mi_network if args.lambda_mi > 0 else None, 
            return_mi_contribution=True
        )
        
        epoch_total_loss += current_loss
        if args.lambda_mi > 0:
            epoch_total_mi_val_train_mi += current_mi_val
            if mi_contribution_to_loss_val is not None:
                epoch_total_mi_contrib_loss += mi_contribution_to_loss_val
        num_batches += 1

        if mi_network and args.lambda_mi > 0:
            pbar.set_postfix_str(f"Loss: {current_loss:.5f}; MI_train: {current_mi_val:.5f}; MI_contrib: {mi_contribution_to_loss_val:.5f}")
        else:
            pbar.set_postfix_str(f"Loss: {current_loss:.5f}")

    avg_epoch_loss = epoch_total_loss / num_batches if num_batches > 0 else float('inf')
    avg_epoch_mi_train = epoch_total_mi_val_train_mi / num_batches if num_batches > 0 and args.lambda_mi > 0 else 0.0
    avg_epoch_mi_contrib = epoch_total_mi_contrib_loss / num_batches if num_batches > 0 and args.lambda_mi > 0 else 0.0

    print(f"Epoch {epoch+1} [Joint Training]: Average Total Loss: {avg_epoch_loss:.5f}")
    if args.lambda_mi > 0:
        print(f"Epoch {epoch+1} [Joint Training]: Average MI_lb from train_mi: {avg_epoch_mi_train:.5f}")
        print(f"Epoch {epoch+1} [Joint Training]: Average MI Contribution to Loss: {avg_epoch_mi_contrib:.5f}")
    return avg_epoch_loss, avg_epoch_mi_train, avg_epoch_mi_contrib


def plot_training_curves(history, phase_name, checkpoint_path, num_epochs):
    epochs_range = range(1, num_epochs + 1)
    
    active_plots = {}
    if history.get('train_loss'): active_plots['loss'] = True
    if history.get('mi_net_mi_lb'): active_plots['mi_phase1'] = True
    if history.get('train_mi_from_train_mi'): active_plots['mi_joint_train'] = True
    if history.get('mi_contrib_loss'): active_plots['mi_joint_contrib'] = True

    num_active_subplots = len(active_plots)
    if num_active_subplots == 0:
        print("No data to plot.")
        return

    plt.figure(figsize=(10, 5 * num_active_subplots))
    current_subplot_idx = 1

    if active_plots.get('loss'):
        plt.subplot(num_active_subplots, 1, current_subplot_idx)
        if history.get('train_loss'):
             plt.plot(epochs_range, history['train_loss'], 'bo-', label='Training Loss')
        if history.get('val_loss'):
             plt.plot(epochs_range, history['val_loss'], 'ro-', label='Validation Loss')
        plt.title(f'{phase_name} - Training & Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        current_subplot_idx += 1
        
    if active_plots.get('mi_phase1'):
        plt.subplot(num_active_subplots, 1, current_subplot_idx)
        plt.plot(epochs_range, history['mi_net_mi_lb'], 'go-', label='Avg MI_lb (MI Net Training)')
        plt.title(f'{phase_name} - MI_lb from MI Net Training')
        plt.xlabel('Epochs')
        plt.ylabel('MI_lb (bits)')
        plt.legend()
        plt.grid(True)
        current_subplot_idx +=1

    if active_plots.get('mi_joint_train'):
        plt.subplot(num_active_subplots, 1, current_subplot_idx)
        plt.plot(epochs_range, history['train_mi_from_train_mi'], 'ms-', label='Avg MI_lb from train_mi (Joint)')
        plt.title(f'{phase_name} - MI_lb from train_mi (Joint Training)')
        plt.xlabel('Epochs')
        plt.ylabel('MI_lb (bits)')
        plt.legend()
        plt.grid(True)
        current_subplot_idx += 1
        
    if active_plots.get('mi_joint_contrib'):
        plt.subplot(num_active_subplots, 1, current_subplot_idx)
        plt.plot(epochs_range, history['mi_contrib_loss'], 'co-', label='Avg MI Contribution to Loss (Joint/Phase2)')
        plt.title(f'{phase_name} - MI Contribution to Total Loss')
        plt.xlabel('Epochs')
        plt.ylabel('MI Loss Term')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(checkpoint_path, f'training_curves_{phase_name.lower().replace(" ", "_").replace(":", "")}.png')
    plt.savefig(save_path)
    print(f"Training curves saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    setup_seed(10) 
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path, exist_ok=True)

    vocab_load_path = args.vocab_file
    if not os.path.isabs(vocab_load_path): 
         base_dir = os.path.dirname(os.path.abspath(__file__))
         vocab_load_path = os.path.join(base_dir, vocab_load_path)
         if not os.path.exists(vocab_load_path) and args.vocab_file.startswith('data/'): 
            vocab_load_path = os.path.join(os.path.dirname(base_dir), args.vocab_file)

    print(f"Loading vocab from: {vocab_load_path}")
    vocab = json.load(open(vocab_load_path, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    
    pad_idx_global = token_to_idx["<PAD>"] 

    deepsc_model = DeepSC(args.num_layers, num_vocab, num_vocab,
                          num_vocab, num_vocab, 
                          args.d_model, args.num_heads,
                          args.dff, 0.1).to(device)
    
    mi_net_model = Mine(hidden_size=10).to(device) 

    criterion_global = nn.CrossEntropyLoss(reduction='none') 
    optimizer_deepsc_global = torch.optim.Adam(deepsc_model.parameters(),
                                 lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    mi_optimizer_global = torch.optim.Adam(mi_net_model.parameters(), lr=1e-3) 

    initNetParams(deepsc_model)
    initNetParams(mi_net_model)

    train_dataset = EurDataset('train')
    val_dataset = EurDataset('test') 
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0,
                                  pin_memory=True, collate_fn=collate_data, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)

    min_val_loss = float('inf')
    
    training_history = {
        'train_loss': [], 'val_loss': [], 'mi_net_mi_lb': [],
        'mi_contrib_loss': [], 'train_mi_from_train_mi': []
    }
    phase_name_str = "Unknown Phase"
    current_epochs_to_run = args.epochs

    if args.training_phase == 1:
        phase_name_str = "Phase 1 MI Net Training"
        current_epochs_to_run = args.epochs_phase1
        print(f"--- Starting {phase_name_str} for {current_epochs_to_run} epochs ---")
        
        if args.deepsc_checkpoint_load_path and os.path.exists(args.deepsc_checkpoint_load_path):
            print(f"Loading DeepSC model from {args.deepsc_checkpoint_load_path} for signal generation.")
            deepsc_model.load_state_dict(torch.load(args.deepsc_checkpoint_load_path, map_location=device))
        
        for param in deepsc_model.parameters(): param.requires_grad = False 
        
        if args.mi_net_checkpoint_load_path and os.path.exists(args.mi_net_checkpoint_load_path):
            print(f"Loading MI Net from {args.mi_net_checkpoint_load_path} to continue training.")
            mi_net_model.load_state_dict(torch.load(args.mi_net_checkpoint_load_path, map_location=device))
        for param in mi_net_model.parameters(): param.requires_grad = True


        for epoch in range(current_epochs_to_run):
            avg_mi_val_epoch = train_phase1_mi_net_only(epoch, args, deepsc_model, mi_net_model, train_dataloader)
            training_history['mi_net_mi_lb'].append(avg_mi_val_epoch)
            
            mi_net_save_path = os.path.join(args.checkpoint_path, f'mi_net_phase1_ep{str(epoch + 1).zfill(2)}.pth')
            torch.save(mi_net_model.state_dict(), mi_net_save_path)
            # print(f"Saved MI Net (Phase 1) model to {mi_net_save_path}")

    elif args.training_phase == 2:
        phase_name_str = "Phase 2 DeepSC Training"
        current_epochs_to_run = args.epochs_phase2
        print(f"--- Starting {phase_name_str} for {current_epochs_to_run} epochs ---")
        
        if not args.mi_net_checkpoint_load_path or not os.path.exists(args.mi_net_checkpoint_load_path):
            raise FileNotFoundError("MI Net checkpoint must be provided for Phase 2 training via --mi_net_checkpoint_load_path")
        
        print(f"Loading pre-trained MI Net from {args.mi_net_checkpoint_load_path}")
        mi_net_model.load_state_dict(torch.load(args.mi_net_checkpoint_load_path, map_location=device))
        for param in mi_net_model.parameters(): param.requires_grad = False 
        
        if args.deepsc_checkpoint_load_path and os.path.exists(args.deepsc_checkpoint_load_path):
            print(f"Loading DeepSC model from {args.deepsc_checkpoint_load_path} to continue training.")
            deepsc_model.load_state_dict(torch.load(args.deepsc_checkpoint_load_path, map_location=device))
        for param in deepsc_model.parameters(): param.requires_grad = True

        for epoch in range(current_epochs_to_run):
            avg_train_loss_epoch, avg_mi_contrib_epoch = train_phase2_deepsc_only(
                epoch, args, deepsc_model, 
                mi_net_model if args.lambda_mi > 0 else None, 
                train_dataloader
            )
            avg_val_loss_epoch = validate(epoch, args, deepsc_model, val_dataloader)
            
            training_history['train_loss'].append(avg_train_loss_epoch)
            training_history['val_loss'].append(avg_val_loss_epoch)
            if args.lambda_mi > 0:
                training_history['mi_contrib_loss'].append(avg_mi_contrib_epoch)
            
            if avg_val_loss_epoch < min_val_loss:
                min_val_loss = avg_val_loss_epoch
                deepsc_save_path = os.path.join(args.checkpoint_path, f'deepsc_phase2_best_ep{str(epoch + 1).zfill(2)}.pth')
                torch.save(deepsc_model.state_dict(), deepsc_save_path)
                print(f"Saved Best DeepSC (Phase 2) model to {deepsc_save_path} (Val Loss: {avg_val_loss_epoch:.4f})")
            
            deepsc_last_save_path = os.path.join(args.checkpoint_path, f'deepsc_phase2_last_ep{str(epoch + 1).zfill(2)}.pth')
            torch.save(deepsc_model.state_dict(), deepsc_last_save_path)

    else: 
        phase_name_str = "Joint Training"
        current_epochs_to_run = args.epochs
        print(f"--- Starting {phase_name_str} for {current_epochs_to_run} epochs ---")
        
        if args.deepsc_checkpoint_load_path and os.path.exists(args.deepsc_checkpoint_load_path):
            print(f"Loading DeepSC model from {args.deepsc_checkpoint_load_path} to continue training.")
            deepsc_model.load_state_dict(torch.load(args.deepsc_checkpoint_load_path, map_location=device))
        if args.mi_net_checkpoint_load_path and os.path.exists(args.mi_net_checkpoint_load_path):
            print(f"Loading MI Net from {args.mi_net_checkpoint_load_path} to continue training.")
            mi_net_model.load_state_dict(torch.load(args.mi_net_checkpoint_load_path, map_location=device))
        for param in deepsc_model.parameters(): param.requires_grad = True
        for param in mi_net_model.parameters(): param.requires_grad = True
            
        for epoch in range(current_epochs_to_run):
            avg_train_loss_epoch, avg_mi_train_epoch, avg_mi_contrib_epoch = train_joint(
                epoch, args, deepsc_model, 
                mi_net_model if args.lambda_mi > 0 else None, 
                train_dataloader
            )
            avg_val_loss_epoch = validate(epoch, args, deepsc_model, val_dataloader)

            training_history['train_loss'].append(avg_train_loss_epoch)
            training_history['val_loss'].append(avg_val_loss_epoch)
            if args.lambda_mi > 0:
                training_history['train_mi_from_train_mi'].append(avg_mi_train_epoch)
                training_history['mi_contrib_loss'].append(avg_mi_contrib_epoch)

            if avg_val_loss_epoch < min_val_loss:
                min_val_loss = avg_val_loss_epoch
                deepsc_save_path = os.path.join(args.checkpoint_path, f'deepsc_joint_best_ep{str(epoch + 1).zfill(2)}.pth')
                torch.save(deepsc_model.state_dict(), deepsc_save_path)
                print(f"Saved Best DeepSC (Joint) model to {deepsc_save_path} (Val Loss: {avg_val_loss_epoch:.4f})")
                if args.lambda_mi > 0:
                    mi_net_save_path = os.path.join(args.checkpoint_path, f'mi_net_joint_best_ep{str(epoch + 1).zfill(2)}.pth')
                    torch.save(mi_net_model.state_dict(), mi_net_save_path)
                    # print(f"Saved MI Net (Joint) model to {mi_net_save_path}")
            
            deepsc_last_save_path = os.path.join(args.checkpoint_path, f'deepsc_joint_last_ep{str(epoch + 1).zfill(2)}.pth')
            torch.save(deepsc_model.state_dict(), deepsc_last_save_path)
            if args.lambda_mi > 0:
                 mi_net_last_save_path = os.path.join(args.checkpoint_path, f'mi_net_joint_last_ep{str(epoch + 1).zfill(2)}.pth')
                 torch.save(mi_net_model.state_dict(), mi_net_last_save_path)

    print("Training finished.")
    if args.plot_curves:
        plot_training_curves(training_history, phase_name_str, args.checkpoint_path, current_epochs_to_run)