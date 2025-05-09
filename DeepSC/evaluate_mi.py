import os
import json
import torch
import argparse
import numpy as np
from dataset import EurDataset, collate_data # Убедитесь, что пути в dataset.py верные
from models.transceiver import DeepSC
from models.mutual_info import Mine, sample_batch, mutual_information
from torch.utils.data import DataLoader
from utils import SNR_to_noise, PowerNormalize, Channels # create_masks, pad_idx, start_idx, end_idx - возможно, тоже понадобятся
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_mi_for_model(model_checkpoint_path, mi_net_checkpoint_path, vocab_file_path, args):
    # --- Загрузка словаря ---
    vocab = json.load(open(vocab_file_path, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]

    # --- Загрузка модели DeepSC ---
    deepsc_model = DeepSC(args.num_layers, num_vocab, num_vocab,
                          num_vocab, num_vocab,
                          args.d_model, args.num_heads,
                          args.dff, 0.1).to(device)
    
    checkpoint_deepsc = torch.load(model_checkpoint_path, map_location=device)
    deepsc_model.load_state_dict(checkpoint_deepsc)
    deepsc_model.eval()
    print(f"DeepSC model loaded from {model_checkpoint_path}")
    
    mi_net = Mine().to(device) # Используйте те же параметры, что и в main.py
    if mi_net_checkpoint_path and os.path.exists(mi_net_checkpoint_path):
        checkpoint_mine = torch.load(mi_net_checkpoint_path, map_location=device)
        mi_net.load_state_dict(checkpoint_mine)
        print(f"Mine model loaded from {mi_net_checkpoint_path}")
    else:
        print(f"Warning: MI Net checkpoint not found at {mi_net_checkpoint_path}. Using freshly initialized MI Net for evaluation (may be incorrect).")

    mi_net.eval()

    # --- Загрузка тестовых данных ---
    test_dataset = EurDataset('test')
    test_iterator = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)

    snr_values_db = np.array([0, 3, 6, 9, 12, 15, 18]) # Как на Рис. 9
    results_mi = []
    channels_util = Channels()

    for snr_db in snr_values_db:
        noise_std_val = SNR_to_noise(snr_db)
        print(f"Evaluating for SNR: {snr_db} dB, Noise Std: {noise_std_val}")
        
        batch_mi_lbs = []
        with torch.no_grad():
            for sents in tqdm(test_iterator, desc=f"SNR {snr_db}dB"):
                sents = sents.to(device)
                
                # Получаем Tx_sig из модели DeepSC
                # Маска для энкодера (хотя для channel_encoder она может быть не нужна напрямую)
                src_mask = (sents == pad_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)
                
                enc_output = deepsc_model.encoder(sents, src_mask)
                channel_enc_output = deepsc_model.channel_encoder(enc_output)
                Tx_sig = PowerNormalize(channel_enc_output) # (batch, seq_len, 16)

                # Симуляция канала (AWGN)
                if args.channel_eval == 'AWGN':
                    Rx_sig = channels_util.AWGN(Tx_sig, noise_std_val)
                elif args.channel_eval == 'Rayleigh':
                     Rx_sig = channels_util.Rayleigh(Tx_sig, noise_std_val)
                else:
                    raise ValueError("Unsupported channel type for evaluation")

                joint, marginal = sample_batch(Tx_sig, Rx_sig)

                mi_lb, _, _ = mutual_information(joint, marginal, mi_net)
                batch_mi_lbs.append(mi_lb.item())
        
        avg_mi_lb = np.mean(batch_mi_lbs)
        results_mi.append(avg_mi_lb)
        print(f"SNR: {snr_db} dB, Average MI: {avg_mi_lb:.4f}")

    return snr_values_db, results_mi


if __name__ == '__main__':
    parser_eval = argparse.ArgumentParser()
    parser_eval.add_argument('--d-model', default=128, type=int)
    parser_eval.add_argument('--dff', default=512, type=int)
    parser_eval.add_argument('--num-layers', default=4, type=int)
    parser_eval.add_argument('--num-heads', default=8, type=int)
    parser_eval.add_argument('--MAX-LENGTH', default=30, type=int)
    parser_eval.add_argument('--batch-size', default=128, type=int) 
    parser_eval.add_argument('--vocab_file', default = 'data/europarl_processed/vocab.json', type=str)
    parser_eval.add_argument('--model-checkpoint-with-mi', default='checkpoints\phase2_deepsc_AWGN_100ep_withMI\deepsc_phase2_best_ep100.pth', type=str, help="Path to DeepSC model trained WITH MI")
    parser_eval.add_argument('--model-checkpoint-without-mi', default='checkpoints\phase0_deepsc_noMI\deepsc_joint_best_ep100.pth', type=str, help="Path to DeepSC model trained WITHOUT MI")
    
    parser_eval.add_argument('--mi-net-checkpoint', default='checkpoints\phase1_mi_net_AWGN_20ep\mi_net_phase1_ep20.pth', type=str, help="Path to trained MI Net model")
    
    parser_eval.add_argument('--channel-eval', default='AWGN', type=str, help="Channel for MI evaluation (AWGN, Rayleigh)")


    args_eval = parser_eval.parse_args()

    print("Evaluating model trained WITH MI component...")
    snr_db_with_mi, mi_values_with_mi = evaluate_mi_for_model(args_eval.model_checkpoint_with_mi, 
                                                              args_eval.mi_net_checkpoint, 
                                                              args_eval.vocab_file, args_eval)

    print("\nEvaluating model trained WITHOUT MI component...")
    snr_db_without_mi, mi_values_without_mi = evaluate_mi_for_model(args_eval.model_checkpoint_without_mi, 
                                                                    args_eval.mi_net_checkpoint, 
                                                                    args_eval.vocab_file, args_eval)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(snr_db_with_mi, mi_values_with_mi, marker='o', label='DeepSC with MI training')
    plt.plot(snr_db_without_mi, mi_values_without_mi, marker='x', label='DeepSC without MI training')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Mutual Information (bits)')
    plt.title('SNR vs Mutual Information (Reproduced)')
    plt.legend()
    plt.grid(True)
    plt.savefig('snr_vs_mi_reproduced.png')
    plt.show()

    print("\nResults with MI training:")
    for snr, mi in zip(snr_db_with_mi, mi_values_with_mi):
        print(f"SNR: {snr} dB, MI: {mi:.4f}")

    print("\nResults without MI training:")
    for snr, mi in zip(snr_db_without_mi, mi_values_without_mi):
        print(f"SNR: {snr} dB, MI: {mi:.4f}")