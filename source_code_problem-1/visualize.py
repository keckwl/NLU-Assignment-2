import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Word groups chosen to show meaningful semantic clusters
WORD_GROUPS = {
    "academic_programs": ["btech", "mtech", "phd", "msc", "mba", "undergraduate", "postgraduate", "degree"],
    "research":          ["research", "publication", "journal", "paper", "project", "lab", "thesis"],
    "people":            ["student", "faculty", "professor", "director", "dean", "researcher", "scholar"],
    "courses":           ["course", "exam", "lecture", "assignment", "syllabus", "credit", "semester"],
    "departments":       ["cse", "ece", "mechanical", "civil", "chemistry", "physics", "mathematics"],
}


def get_word_vectors(model, word_groups):
    vectors, labels, group_ids = [], [], []
    for group_idx, (_, words) in enumerate(word_groups.items()):
        for word in words:
            if word in model.wv:
                vectors.append(model.wv[word])
                labels.append(word)
                group_ids.append(group_idx)
    if not vectors:
        return None, None, None
    return np.array(vectors), labels, group_ids


def plot_embeddings(reduced, labels, group_ids, group_names, title, output_path):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    palette = cm.get_cmap("tab10")
    colors  = [palette(gid) for gid in group_ids]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(reduced[:, 0], reduced[:, 1], c=colors, s=80, alpha=0.8,
               edgecolors="k", linewidths=0.4)
    for i, word in enumerate(labels):
        ax.annotate(word, xy=(reduced[i, 0], reduced[i, 1]),
                    xytext=(4, 4), textcoords="offset points", fontsize=8)

    legend_patches = [mpatches.Patch(color=palette(i), label=name)
                      for i, name in enumerate(group_names)]
    ax.legend(handles=legend_patches, loc="best", fontsize=9)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def visualize_model(model, model_name, output_dir="outputs"):
    vectors, labels, group_ids = get_word_vectors(model, WORD_GROUPS)
    if vectors is None or len(vectors) < 5:
        return

    group_names = list(WORD_GROUPS.keys())

    # PCA projection
    pca_reduced = PCA(n_components=2, random_state=42).fit_transform(vectors)
    plot_embeddings(pca_reduced, labels, group_ids, group_names,
                    title=f"PCA Projection [{model_name}]",
                    output_path=os.path.join(output_dir, f"pca_{model_name.lower()}.png"))

    # t-SNE projection; perplexity must be < n_samples
    perplexity = min(30, len(vectors) - 1)
    try:
        tsne_reduced = TSNE(n_components=2, perplexity=perplexity,
                            random_state=42, max_iter=1000).fit_transform(vectors)
    except TypeError:
        # max_iter renamed from n_iter in scikit-learn >= 1.4
        tsne_reduced = TSNE(n_components=2, perplexity=perplexity,
                            random_state=42, n_iter=1000).fit_transform(vectors)
    plot_embeddings(tsne_reduced, labels, group_ids, group_names,
                    title=f"t-SNE Projection [{model_name}]",
                    output_path=os.path.join(output_dir, f"tsne_{model_name.lower()}.png"))


def run_visualization(model_dir="outputs/models", output_dir="outputs"):
    cbow_model = Word2Vec.load(os.path.join(model_dir, "cbow_dim100_win5_neg5.model"))
    sg_model   = Word2Vec.load(os.path.join(model_dir, "skipgram_dim100_win5_neg5.model"))
    visualize_model(cbow_model, "CBOW", output_dir)
    visualize_model(sg_model,   "Skipgram", output_dir)


if __name__ == "__main__":
    run_visualization()
