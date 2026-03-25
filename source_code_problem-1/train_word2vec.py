import os
import json
import itertools
from gensim.models import Word2Vec


def load_tokenized_corpus(clean_path="clean_corpus.txt"):
    sentences = []
    with open(clean_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                sentences.append(tokens)
    return sentences


def train_model(sentences, sg, vector_size, window, negative, epochs=10, min_count=2):
    # sg=0 -> CBOW, sg=1 -> Skip-gram; hs=0 uses negative sampling
    return Word2Vec(
        sentences=sentences, vector_size=vector_size, window=window,
        negative=negative, sg=sg, hs=0, min_count=min_count,
        workers=4, epochs=epochs, seed=42,
    )


def run_experiments(sentences, output_dir="outputs/models"):
    os.makedirs(output_dir, exist_ok=True)

    # Hyperparameter grid: embedding dims, window sizes, negative samples
    embedding_dims   = [50, 100, 200]
    window_sizes     = [3, 5]
    negative_samples = [5, 10]

    results = []
    for model_name, sg_flag in {"cbow": 0, "skipgram": 1}.items():
        for dim, win, neg in itertools.product(embedding_dims, window_sizes, negative_samples):
            config_str = f"{model_name}_dim{dim}_win{win}_neg{neg}"
            model = train_model(sentences, sg=sg_flag, vector_size=dim, window=win, negative=neg)
            model_path = os.path.join(output_dir, f"{config_str}.model")
            model.save(model_path)
            results.append({
                "model_type": model_name, "sg": sg_flag,
                "vector_size": dim, "window": win, "negative": neg,
                "vocab_size": len(model.wv), "model_path": model_path,
            })

    with open(os.path.join("outputs", "experiment_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    sentences = load_tokenized_corpus("clean_corpus.txt")
    run_experiments(sentences)
