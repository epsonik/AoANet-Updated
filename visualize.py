"""
Skrypt do wizualizacji wag atencji, w tym dla tokenu <eos>.
Użycie:
python visualize.py --model_path /path/to/your/model.pth --image_path /path/to/your/image.jpg --infos_path /path/to/infos.pkl
"""

import torch
import misc.utils as utils
import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as trn


def main(opt):
    # --- Ładowanie modelu i informacji ---
    if not os.path.isfile(opt.model_path):
        raise FileNotFoundError(f"Model file not found: {opt.model_path}")
    if not os.path.isfile(opt.infos_path):
        raise FileNotFoundError(f"Infos file not found: {opt.infos_path}")

    # Załaduj infos, które zawiera słownik (vocab) i opcje modelu
    with open(opt.infos_path, 'rb') as f:
        infos = utils.pickle_load(f)

    # Zastąp opcje z infos opcjami z linii komend, jeśli podano
    for k, v in vars(opt).items():
        if v is not None:
            vars(infos['opt'])[k] = v

    vocab = infos['vocab']
    model_opt = infos['opt']

    # Załaduj model
    model = utils.setup(model_opt)
    model.load_state_dict(torch.load(opt.model_path, map_location=torch.device('cpu')))
    model.eval()  # Ustaw model w tryb ewaluacji

    # --- Przygotowanie obrazu ---
    if not os.path.isfile(opt.image_path):
        raise FileNotFoundError(f"Image file not found: {opt.image_path}")

    # Transformacje obrazu muszą pasować do tych używanych podczas treningu
    # Zazwyczaj jest to resize, a następnie normalizacja
    preprocess = trn.Compose([
        trn.Resize((model_opt.image_crop_size, model_opt.image_crop_size)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_pil = Image.open(opt.image_path).convert('RGB')
    img_tensor = preprocess(img_pil).unsqueeze(0)  # Dodaj wymiar batcha

    # --- Generowanie opisu i wag atencji ---
    with torch.no_grad():
        # Wywołaj sample z nową opcją `return_att_weights`
        # Model powinien zwrócić 3 wartości: seq, seqLogprobs, att_weights
        seq, _, att_weights = model(img_tensor, None, opt={'beam_size': 1, 'return_att_weights': True})

    # --- Wizualizacja ---
    # Zdekoduj sekwencję na słowa
    sents = utils.decode_sequence(vocab, seq)
    sent = sents[0]
    words = sent.split(' ')

    # ZNAJĎ INDEKS <eos> i utnij słowa do tego miejsca
    # seq jest tensorem [1, seq_length]
    try:
        eos_index = (seq[0] == 0).nonzero(as_tuple=False)[0][0].item()
        words = words[:eos_index]
    except IndexError:
        # <eos> nie zostało wygenerowane, użyj wszystkich słów
        eos_index = len(words)

    # RĘCZNIE DODAJ TOKEN <eos> do listy słów
    words.append('<eos>')

    # Pobierz wagi atencji dla pierwszego (i jedynego) obrazu
    # Kształt: (1, seq_length, num_regions) -> (seq_length, num_regions)
    att_weights = att_weights[0]

    # Ogranicz wagi do wygenerowanej długości + 1 (dla <eos>)
    att_weights = att_weights[:eos_index + 1]

    # Sprawdzenie spójności
    print(f"Wygenerowane zdanie: '{' '.join(words[:-1])}'")
    print(f"Słowa do wizualizacji ({len(words)}): {words}")
    print(f"Liczba map atencji: {att_weights.size(0)}")

    if len(words) != att_weights.size(0):
        print("OSTRZEŻENIE: Liczba słów nie zgadza się z liczbą map atencji. Wizualizacja może być niekompletna.")
        # Ogranicz do mniejszej z dwóch wartości
        min_len = min(len(words), att_weights.size(0))
        words = words[:min_len]
        att_weights = att_weights[:min_len]

    # Rozmiar siatki atencji (np. 14x14 = 196)
    # W modelach AoA, atencja jest nad regionami + 1 "fake region", więc odejmujemy 1
    num_regions = att_weights.size(1)
    # Sprawdź, czy atencja jest nad siatką, czy listą regionów
    att_grid_size = int(np.sqrt(num_regions - 1))  # np. sqrt(196) = 14
    if (att_grid_size * att_grid_size) != (num_regions - 1):
        raise ValueError(f"Nie można przekształcić atencji o rozmiarze {num_regions - 1} na kwadratową siatkę.")

    # Stwórz katalog na wyniki, jeśli nie istnieje
    output_dir = "attention_viz"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Zapisywanie wizualizacji w katalogu: {output_dir}")

    # Pętla wizualizująca
    for i in range(len(words)):
        word = words[i]

        # Pobierz mapę atencji bez "fake region" i przekształć na siatkę 2D
        attention_map = att_weights[i][1:].view(att_grid_size, att_grid_size).numpy()

        # Utwórz figurę
        fig, ax = plt.subplots()
        ax.imshow(img_pil)

        # Nałóż mapę atencji
        # Użyj `resize` z Pillow, aby poprawnie przeskalować mapę
        alpha_img = Image.fromarray((attention_map / attention_map.max() * 255).astype(np.uint8))
        alpha_img = alpha_img.resize(img_pil.size, Image.Resampling.LANCZOS)
        ax.imshow(np.array(alpha_img), cmap='jet', alpha=0.5)

        ax.set_title(f"Atencja dla słowa: '{word}'", fontsize=14)
        ax.axis('off')

        # Zapisz plik
        filename = os.path.join(output_dir, f"{i:02d}_attention_{word.replace('<', '_').replace('>', '_')}.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        print(f"Zapisano: {filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Wizualizacja wag atencji modelu.")
    parser.add_argument('--model_path', type=str, required=True, help='Ścieżka do zapisanego modelu (.pth).')
    parser.add_argument('--infos_path', type=str, required=True, help='Ścieżka do pliku infos z treningu (.pkl).')
    parser.add_argument('--image_path', type=str, required=True, help='Ścieżka do obrazu do analizy.')

    # Opcje, które można nadpisać
    parser.add_argument('--image_crop_size', type=int, default=224, help='Rozmiar, do którego przycinany jest obraz.')

    args = parser.parse_args()
    main(args)