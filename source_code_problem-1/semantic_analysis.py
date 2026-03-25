import os
import json
from gensim.models import Word2Vec


def load_model(path):
    return Word2Vec.load(path)


def get_nearest_neighbors(model, words, topn=5):
    # Cosine similarity lookup for each query word
    results = {}
    for word in words:
        results[word] = model.wv.most_similar(word, topn=topn) if word in model.wv else []
    return results


def run_analogy(model, word_a, word_b, word_c, topn=5):
    # Solves word_a:word_b :: word_c:? via vector arithmetic
    try:
        return model.wv.most_similar(positive=[word_b, word_c], negative=[word_a], topn=topn)
    except KeyError:
        return []


def print_neighbors(results, model_name):
    print(f"\nNearest Neighbors [{model_name}]")
    for word, neighbors in results.items():
        if neighbors:
            print(f"  '{word}':")
            for rank, (neighbor, score) in enumerate(neighbors, 1):
                print(f"    {rank}. {neighbor:<20} (cosine sim: {score:.4f})")
        else:
            print(f"  '{word}' -> NOT IN VOCABULARY")


def print_analogies(analogy_results, model_name):
    print(f"\nAnalogy Results [{model_name}]")
    for (a, b, c), answers in analogy_results.items():
        print(f"  {a} : {b} :: {c} : ?")
        if answers:
            for rank, (word, score) in enumerate(answers[:3], 1):
                print(f"    {rank}. {word:<20} (score: {score:.4f})")
        else:
            print("    -> No result (words not in vocab)")


def run_semantic_analysis(model_dir="outputs/models", output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    cbow_model = load_model(os.path.join(model_dir, "cbow_dim100_win5_neg5.model"))
    sg_model   = load_model(os.path.join(model_dir, "skipgram_dim100_win5_neg5.model"))

    query_words = ["research", "student", "phd", "exam", "faculty"]
    cbow_neighbors = get_nearest_neighbors(cbow_model, query_words)
    sg_neighbors   = get_nearest_neighbors(sg_model,   query_words)

    print_neighbors(cbow_neighbors, "CBOW")
    print_neighbors(sg_neighbors,   "Skip-gram")

    # Analogy experiments using words confirmed in vocabulary
    analogies = [
        ("mtech", "program",   "phd"),        # mtech:program :: phd:?
        ("professor", "research", "student"), # professor:research :: student:?
        ("semester", "course",  "program"),   # semester:course :: program:?
    ]
    cbow_analogies = {(a, b, c): run_analogy(cbow_model, a, b, c) for a, b, c in analogies}
    sg_analogies   = {(a, b, c): run_analogy(sg_model,   a, b, c) for a, b, c in analogies}

    print_analogies(cbow_analogies, "CBOW")
    print_analogies(sg_analogies,   "Skip-gram")

    all_results = {
        "cbow_neighbors": {w: v for w, v in cbow_neighbors.items()},
        "sg_neighbors":   {w: v for w, v in sg_neighbors.items()},
        "cbow_analogies": {f"{a}:{b}::{c}": v for (a, b, c), v in cbow_analogies.items()},
        "sg_analogies":   {f"{a}:{b}::{c}": v for (a, b, c), v in sg_analogies.items()},
    }
    with open(os.path.join(output_dir, "semantic_analysis.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    return cbow_model, sg_model


if __name__ == "__main__":
    run_semantic_analysis()
