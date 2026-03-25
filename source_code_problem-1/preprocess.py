import re
import os
import json

import nltk
from nltk.tokenize import word_tokenize

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

import matplotlib.pyplot as plt

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def load_raw_corpus(path="raw_corpus.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def preprocess_document(text):
    # Lowercase, remove non-alpha chars, tokenize, keep alpha tokens >= 2 chars
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return [t for t in word_tokenize(text) if t.isalpha() and len(t) >= 2]


def preprocess_corpus(raw_docs):
    tokenized_docs = [preprocess_document(doc) for doc in raw_docs if preprocess_document(doc)]
    flat_tokens = [tok for doc in tokenized_docs for tok in doc]
    return tokenized_docs, flat_tokens


def report_statistics(tokenized_docs, flat_tokens):
    stats = {
        "num_documents": len(tokenized_docs),
        "total_tokens": len(flat_tokens),
        "vocabulary_size": len(set(flat_tokens)),
    }
    print(f"Documents: {stats['num_documents']} | Tokens: {stats['total_tokens']} | Vocab: {stats['vocabulary_size']}")
    return stats


def generate_wordcloud(flat_tokens, output_path="outputs/wordcloud.png"):
    if not WORDCLOUD_AVAILABLE:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wc = WordCloud(width=1200, height=600, background_color="white",
                   max_words=150, colormap="viridis").generate(" ".join(flat_tokens))
    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Most Frequent Words in IIT Jodhpur Corpus", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_clean_corpus(tokenized_docs, output_path="clean_corpus.txt"):
    # Each line is space-separated tokens for one document (Gensim Word2Vec format)
    with open(output_path, "w", encoding="utf-8") as f:
        for tokens in tokenized_docs:
            f.write(" ".join(tokens) + "\n")


def run_preprocessing(raw_path="raw_corpus.txt", clean_path="clean_corpus.txt"):
    raw_docs = load_raw_corpus(raw_path)
    tokenized_docs, flat_tokens = preprocess_corpus(raw_docs)
    stats = report_statistics(tokenized_docs, flat_tokens)
    generate_wordcloud(flat_tokens, output_path="outputs/wordcloud.png")
    save_clean_corpus(tokenized_docs, output_path=clean_path)
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return tokenized_docs, flat_tokens, stats


if __name__ == "__main__":
    run_preprocessing()
