from scrape_data import collect_corpus
from preprocess import run_preprocessing
from train_word2vec import load_tokenized_corpus, run_experiments
from semantic_analysis import run_semantic_analysis
from visualize import run_visualization

collect_corpus(output_path="raw_corpus.txt")
run_preprocessing(raw_path="raw_corpus.txt", clean_path="clean_corpus.txt")
sentences = load_tokenized_corpus("clean_corpus.txt")
run_experiments(sentences, output_dir="outputs/models")
run_semantic_analysis(model_dir="outputs/models", output_dir="outputs")
run_visualization(model_dir="outputs/models", output_dir="outputs")
