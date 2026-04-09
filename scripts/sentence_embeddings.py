import argparse
import pandas as pd
import numpy as np
from gensim.models import FastText
from collections import Counter
from sklearn.decomposition import TruncatedSVD


def load_sentences_and_labels(files):
    """
    THe function reads the TSV files and returns
    a list of sentences and a list of labels.
    """
    sentences = []
    labels = []

    for file in files:
        df = pd.read_csv(file, sep='\t')

        for idx, row in df.iterrows():
            sentences.append(list(str(row['text'])))
            labels.append(row['category'])

    return sentences, labels


def sentence_embeddings(sentences, model, use_sif=False, freq=None, total=None):
    """
    Converting sentences to embeddings.
    If use_sif=True, use of weighted averaging.
    """

    embeddings = []

    for sent in sentences:
        char_vectors = []

        for char in sent:
            if char in model.wv:
                
                #if SIF enabled, compute a weight for this character
                if use_sif:
                    a = 1e-3
                    p = freq[char] / total
                    weight = a / (a + p)
                # else all characters are equal
                else:
                    weight = 1.0

                char_vectors.append(weight * model.wv[char])

        if char_vectors:
            sent_vec = np.mean(char_vectors, axis=0)
        else:
            sent_vec = np.zeros(model.vector_size)

        embeddings.append(sent_vec)

    return np.array(embeddings)


def main():
    parser = argparse.ArgumentParser(description="Compute sentence embeddings")

    parser.add_argument('--input_files', nargs='+', required=True, help="TSV files")
    parser.add_argument('--fasttext_model', required=True, help="Path to FastText model")
    parser.add_argument('--output_file', required=True, help="Output .npz file")
    parser.add_argument('--use_sif', action='store_true', help="Use SIF weighting")
    args = parser.parse_args()

    # load data
    sentences, labels = load_sentences_and_labels(args.input_files)

    # load FastText
    model = FastText.load(args.fasttext_model)

    # if using SIF, count the frequency of each character
    if args.use_sif:
        freq = Counter()
        for sent in sentences:
            for char in sent:
                freq[char] += 1
        total = sum(freq.values())
    else:
        freq = None
        total = None

    # compute embeddings
    embeddings = sentence_embeddings(
        sentences,
        model,
        use_sif=args.use_sif,
        freq=freq,
        total=total
    )

    if args.use_sif:
        svd = TruncatedSVD(n_components=1)
        svd.fit(embeddings)
        pc = svd.components_

        embeddings = embeddings - embeddings.dot(pc.T) * pc

    np.savez(args.output_file, embeddings=embeddings, labels=np.array(labels))

    print(f"Saved {len(sentences)} sentence embeddings to {args.output_file}")


if __name__ == '__main__':
    main()