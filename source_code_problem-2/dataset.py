import torch
from torch.utils.data import Dataset


def load_names(path="TrainingNames.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]


def build_vocab(names):
    # Special tokens: <PAD>=0, <SOS>=1, <EOS>=2
    chars = sorted(set("".join(names)))
    char2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
    for ch in chars:
        if ch not in char2idx:
            char2idx[ch] = len(char2idx)
    idx2char = {v: k for k, v in char2idx.items()}
    return char2idx, idx2char


def encode_name(name, char2idx):
    # Wrap name with SOS and EOS tokens
    return ([char2idx["<SOS>"]]
            + [char2idx[ch] for ch in name if ch in char2idx]
            + [char2idx["<EOS>"]])


def decode_sequence(indices, idx2char):
    # Convert indices back to string, stopping at EOS
    chars = []
    for idx in indices:
        token = idx2char.get(idx, "")
        if token == "<EOS>":
            break
        if token not in ("<SOS>", "<PAD>"):
            chars.append(token)
    return "".join(chars)


class NamesDataset(Dataset):
    # Teacher forcing: input=[SOS,c1..cn], target=[c1..cn,EOS]
    def __init__(self, names, char2idx):
        self.encoded = [encode_name(name, char2idx) for name in names]

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        seq = self.encoded[idx]
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)


def collate_fn(batch):
    # Pad variable-length sequences to the longest in the batch
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    xs_padded = torch.zeros(len(xs), max_len, dtype=torch.long)
    ys_padded = torch.zeros(len(ys), max_len, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        xs_padded[i, :x.size(0)] = x
        ys_padded[i, :y.size(0)] = y
    return xs_padded, ys_padded
