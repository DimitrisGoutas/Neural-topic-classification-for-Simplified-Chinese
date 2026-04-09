import argparse
import pandas as pd
from gensim.models import FastText

def read_sentences_from_tsv(files):
    """ 
    Reads all TSV files and returns a list of sentences.
    Each sentence will be a list of characters. 
    """
    sentences = []
    
    for file in files:
        # read the TSV files
        df = pd.read_csv(file, sep='\t')
        for text in df['text']:
            # convert sentence to a list of characters
            chars = list(str(text)) 
            sentences.append(chars) 

    return sentences

def main():
    #use of argparse to allow command-line scripts
    parser = argparse.ArgumentParser(description="Train FastText embeddings on Chinese dataset")
    parser.add_argument('--input_files', nargs='+', required=True, help="Choose TSV files")
    parser.add_argument('--embedding_dim', type=int, default=100, help="Dimension of embeddings")
    parser.add_argument('--output_file', type=str, required=True, help="Create a file to save the model")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    args = parser.parse_args()

    sentences = read_sentences_from_tsv(args.input_files)

    #train fasttext embeddings 
    model = FastText(
        sentences= sentences,      
        vector_size=args.embedding_dim,
        window= 5,
        min_count=1,
        sg= 1,
        epochs= args.epochs
    )

    #save the model to the output file
    model.save(args.output_file)
    print(f"The model has been saved to {args.output_file}")

if __name__ == '__main__':
    main()