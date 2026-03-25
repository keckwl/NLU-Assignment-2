import os
import json
import torch

from dataset import load_names, decode_sequence
from models  import VanillaRNN, BidirectionalLSTM, RNNWithAttention

EMBED_DIM   = 32
HIDDEN_SIZE = 64
NUM_LAYERS  = 1


def load_vocab(output_dir="outputs"):
    with open(os.path.join(output_dir, "vocab.json"), "r") as f:
        data = json.load(f)
    return data["char2idx"], {int(k): v for k, v in data["idx2char"].items()}


def load_model(model_class, vocab_size, ckpt_path, device):
    model = model_class(vocab_size=vocab_size, embed_dim=EMBED_DIM,
                        hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=0.0)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.to(device).eval()


def generate_names(model, char2idx, idx2char, n=200, max_len=20, temperature=1.2, device="cpu"):
    sos_idx, eos_idx = char2idx["<SOS>"], char2idx["<EOS>"]
    names = []
    for _ in range(n):
        indices = model.generate(sos_idx=sos_idx, eos_idx=eos_idx,
                                 max_len=max_len, temperature=temperature, device=device)
        name = decode_sequence(indices, idx2char)
        # Filter out degenerate outputs where >60% of chars are the same
        if name and len(name) >= 2:
            if max(name.count(c) for c in set(name)) / len(name) < 0.6:
                names.append(name)
    return names


def compute_novelty_rate(generated_names, training_names):
    # Fraction of generated names not present in the training set
    training_set = set(n.lower() for n in training_names)
    novel = sum(1 for n in generated_names if n.lower() not in training_set)
    return novel / len(generated_names) if generated_names else 0.0


def compute_diversity(generated_names):
    # Unique names/total generated names
    if not generated_names:
        return 0.0
    return len(set(generated_names)) / len(generated_names)


def evaluate_all(names_path="TrainingNames.txt", output_dir="outputs"):
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_names = load_names(names_path)
    char2idx, idx2char = load_vocab(output_dir)
    vocab_size     = len(char2idx)

    model_registry = {
        "VanillaRNN"      : VanillaRNN,
        "BLSTM"           : BidirectionalLSTM,
        "RNNWithAttention": RNNWithAttention,
    }

    all_results, all_samples = {}, {}

    for model_name, model_class in model_registry.items():
        ckpt_path = os.path.join(output_dir, f"{model_name}_best.pt")
        if not os.path.exists(ckpt_path):
            continue

        model     = load_model(model_class, vocab_size, ckpt_path, device)
        generated = generate_names(model, char2idx, idx2char, device=device)
        novelty   = compute_novelty_rate(generated, training_names)
        diversity = compute_diversity(generated)

        print(f"\nModel: {model_name}")
        print(f"  Params    : {model.count_parameters():,}")
        print(f"  Novelty   : {novelty*100:.1f}%")
        print(f"  Diversity : {diversity*100:.1f}%")
        print(f"  Samples   : {', '.join(generated[:10])}")

        all_results[model_name] = {"novelty_rate": novelty, "diversity": diversity,
                                   "num_params": model.count_parameters()}
        all_samples[model_name] = generated[:50]

        # Free memory after each model
        del model
        torch.cuda.empty_cache()

    # Qualitative analysis
    print("\nQualitative Analysis:")
    print("  VanillaRNN      — phonetically plausible names; limited long-range memory due to vanishing gradients.")
    print("  BLSTM           — bidirectional context captures richer patterns; may produce short fragments on small data.")
    print("  RNNWithAttention — attention improves structural coherence; can struggle with very small training sets.")

    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    with open(os.path.join(output_dir, "generated_samples.json"), "w") as f:
        json.dump(all_samples, f, indent=2)

    return all_results


if __name__ == "__main__":
    evaluate_all()
