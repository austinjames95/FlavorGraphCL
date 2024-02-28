import torch
from parser_1 import parameter_parser
from utils import tab_printer, graph_reader, evaluate
from dataloader import DataReader, DatasetLoader
from graph2vec import Metapath2Vec, Node2Vec
from plotter import plot_embedding
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    """
    Parsing command line parameters.
    Reading data, embedding base graph, creating persona graph and learning a splitter.
    Saving the persona mapping and the embedding.
    """
    args = parameter_parser()
    torch.manual_seed(args.seed)
    tab_printer(args)

    """
    1. read graph and load as torch dataset
    """
    graph, graph_ingr_only = graph_reader(args.input_nodes, args.input_edges)


    """
    2. Metapath2vec with MetaPathWalker - Ingredient-Ingredient / Ingredient-Food-like Compound / Ingredient-Drug-like Compound
    """
    
    if args.idx_embed == 'Node2vec':
        node2vec = Node2Vec(args, graph)
        node2vec.train()
        embeddings = node2vec.get_embeddings()

    else:
        metapath2vec = Metapath2Vec(args, graph)
        metapath2vec.train()
        embeddings = metapath2vec.get_embeddings()


    export_embeddings_to_csv(embeddings, args.output_file)
    """
    3. Plot your embedding if you like
    """
    plot_embedding(args, graph)

    """
    4. Evaluate Node Classification & Node Clustering
    """
    evaluate(args, graph)

def export_embeddings_to_csv(embeddings, output_file):
    np.savetxt(output_file, embeddings, delimiter=",")

if __name__ == "__main__":
    main()
 