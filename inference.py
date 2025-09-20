import argparse
import json
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import skimage.transform

import models
from dataloaderraw import ImageLoader
from misc import utils


# Krok 1: Zmodyfikowana funkcja próbkowania do zwracania wag uwagi
def sample_with_attention(model, img, opt={}):
    """
    Zmodyfikowana wersja metody _sample z modelu AttModel,
    która zwraca również wagi uwagi.
    """
    beam_size = opt.get('beam_size', 10)

    # enkodowanie obrazu
    fc_feats, att_feats, _ = model.cnn(img)
    fc_feats = fc_feats.expand(beam_size, model.fc_feat_size)
    att_feats = att_feats.expand(beam_size, model.att_num, model.att_feat_size)

    # Inicjalizacja stanu
    state = model.init_hidden(beam_size)

    # Gotowe sekwencje i ich wyniki
    done_beams = []

    # Inicjalizacja wiązki
    # Słownik: 'seq', 'logps', 'p_seq', 'state', 'att'
    beams = [{'seq': torch.zeros((1, 0), dtype=torch.long).cuda(), 'logps': torch.zeros(1).cuda()}]

    for t in range(model.seq_length + 1):
        all_inds = []
        all_logps = []
        all_atts = []  # Przechowywanie wag uwagi

        for beam in beams:
            if t > 0:
                last_word = beam['seq'][:, -1]
                if last_word.item() == 0:  # Koniec sekwencji
                    done_beams.append(beam)
                    continue

            # Przygotowanie wejścia
            xt = model.embed(last_word.unsqueeze(0)) if t > 0 else model.embed(
                torch.zeros(1, 1, dtype=torch.long).cuda())

            # Obliczenie wyjścia i wag uwagi
            out, state, att = model.core(xt, fc_feats, att_feats, state)
            logprobs = torch.nn.functional.log_softmax(model.logit(out), dim=1)

            # Rozszerzenie wiązki
            logps, inds = torch.topk(logprobs, beam_size, dim=1)

            all_inds.append(inds)
            all_logps.append(logps + beam['logps'].unsqueeze(1))
            all_atts.append(att)

        if not all_inds: break

        # Stworzenie nowych wiązek
        new_beams = []
        all_logps = torch.cat(all_logps)
        all_inds = torch.cat(all_inds)
        all_atts = torch.cat(all_atts)

        top_logps, top_inds = torch.topk(all_logps.view(-1), beam_size)

        for i in range(beam_size):
            logp = top_logps[i]
            beam_index = top_inds[i] // beam_size
            word_index = top_inds[i] % beam_size

            original_beam = beams[beam_index]
            new_seq = torch.cat([original_beam['seq'], all_inds[beam_index, word_index].unsqueeze(0).unsqueeze(0)],
                                dim=1)

            new_beam = {
                'seq': new_seq,
                'logps': logp,
                'att': original_beam.get('att', []) + [all_atts[beam_index]]
            }
            new_beams.append(new_beam)

        beams = new_beams

    # Wybór najlepszej wiązki
    if not done_beams:
        done_beams = beams

    best_beam = sorted(done_beams, key=lambda x: x['logps'], reverse=True)[0]

    return best_beam['seq'], best_beam['att']


# Krok 2: Funkcja do wizualizacji uwagi
def visualize_attention_aoa(image_path, seq, att_weights, vocab, smooth=True):
    """
    Wizualizuje uwagę dla modelu AoANet.
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [vocab.get(str(ix.item()), '') for ix in seq[0]]
    words = [word for word in words if word not in ['<start>', '<end>', '<pad>', '']]

    num_words = len(words)
    w = np.ceil(np.sqrt(num_words + 1))
    h = np.ceil((num_words + 1) / w)

    plt.figure(figsize=(12, 12))

    # Oryginalny obraz
    plt.subplot(h, w, 1)
    plt.imshow(image)
    plt.axis('off')

    for i, (word, att) in enumerate(zip(words, att_weights)):
        if i >= len(att_weights): break

        plt.subplot(h, w, i + 2)
        plt.text(0, 1, '%s' % (word), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)

        # Przekształcenie wektora uwagi na siatkę (np. 6x6 dla 36 regionów)
        att_dim = int(np.sqrt(att.shape[1]))
        alpha = att.cpu().numpy().reshape(att_dim, att_dim)

        if smooth:
            alpha = skimage.transform.pyramid_expand(alpha, upscale=24, sigma=8)

        plt.imshow(alpha, alpha=0.7, cmap='jet')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('attention_visualization.png')
    print("Wizualizacja uwagi została zapisana w pliku: attention_visualization.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Ścieżka do wytrenowanego modelu (pliku .pth).')
    parser.add_argument('--infos_path', type=str, required=True, help='Ścieżka do pliku infos (infos.pkl).')
    parser.add_argument('--image_path', type=str, required=True, help='Ścieżka do obrazu do przetworzenia.')
    parser.add_argument('--beam_size', type=int, default=5, help='Rozmiar wiązki dla wyszukiwania wiązkowego.')

    # Dodaj inne opcje, które mogą być potrzebne
    parser.add_argument('--rnn_size', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--drop_prob_lm', type=float, default=0.5)
    parser.add_argument('--seq_length', type=int, default=16)
    parser.add_argument('--fc_feat_size', type=int, default=2048)
    parser.add_argument('--att_feat_size', type=int, default=2048)
    parser.add_argument('--att_hid_size', type=int, default=512)
    parser.add_argument('--use_bn', type=int, default=0)

    opt = parser.parse_args()
    vars(opt).update({'input_json': '', 'input_fc_dir': '', 'input_att_dir': '', 'input_label_h5': ''})

    # Wczytaj informacje o modelu i słownictwie
    with open(opt.infos_path, 'rb') as f:
        infos = utils.pickle_load(f)

    # Zastąp opcje z wiersza poleceń opcjami z infos
    for k, v in infos['opt'].__dict__.items():
        if k not in vars(opt):
            vars(opt).update({k: v})

    opt.vocab = infos['vocab']

    # Wczytaj model
    model = models.setup(opt)
    model.load_state_dict(torch.load(opt.model))
    model.cuda()
    model.eval()

    # Wczytaj i przetwórz obraz
    loader = ImageLoader({'dataroot': ''})
    img, _ = loader.get_image(opt.image_path)
    img = img.unsqueeze(0).cuda()

    # Krok 3: Wygeneruj podpis i wagi uwagi
    seq, att_weights = sample_with_attention(model, img, vars(opt))

    # Zdekoduj sekwencję na słowa
    sents = utils.decode_sequence(opt.vocab, seq)
    print("Wygenerowany podpis:", sents[0])

    # Wywołaj funkcję wizualizacji
    visualize_attention_aoa(opt.image_path, seq, att_weights, opt.vocab)