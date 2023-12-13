# from embedding import Embed,EmbedBERT

import warnings
#ignoring sklearn deprecation warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn

from bef.search_utils import (vector_search, 
                    search_and_print)
import numpy as np
import faiss
import pickle5 as pickle
import torch
import os
import argparse
import json
from sentence_transformers import SentenceTransformer

from faiss_utils.faiss_indexes import (create_index_flat, create_index_HNSW,
                            create_index_compressed,create_index_IVF)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, default='../data/pubmed_small_graph/',
                        help="The directory where embeddings.json exists ")

    parser.add_argument("--embs_file", type=str, default='sentence_bert_embs_graph.json',
                        help="The embeddings file")

    parser.add_argument("--indexname", type=str, default='faiss.bin',
                        help="The name for the output index file")

    return parser.parse_args()


def main(args):

    DATA_DIR = args.indir  
    embeddings = []
    print('++++++LOADING EMBEDDINGS==============')
    with open(DATA_DIR + args.embs_file) as ff:
        for g in json.load(ff):
            embeddings.append(g['embedding'])

    #TypeError: in method 'IndexPreTransform_train', argument 3 of type 'float const *'
    embeddings = np.array(embeddings, dtype=np.float32)
    d = embeddings.shape[1]

    print('++++++CREATING FAISS INDEX==============')

    index = create_index_flat(embeddings, d)

    # index_file = os.path.join('out', ONTOLOGY, ONTOLOGY+'.index')
    faiss.write_index(index, os.path.join(args.indir,args.indexname ) )



if __name__ == '__main__':

    main(parse_args())
