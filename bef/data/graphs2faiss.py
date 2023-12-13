
import faiss
import warnings
#ignoring sklearn deprecation warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

import os
import numpy as np
import pickle
import json
import torch
import argparse
from sentence_transformers import SentenceTransformer

from faiss_utils.faiss_indexes import create_index_flat,create_index_compressed
from bef.search.vector_utils import compute_kernel_bias,transform_and_normalize
from networkx.readwrite import json_graph
from bef.data.graph_deduplicator import graph_to_hashable
from tqdm import tqdm

class EventsDataset(torch.utils.data.Dataset):

  def __init__(self, graphs):
      'Initialization'
      self.graphs = graphs

  def __len__(self):
      'Denotes the total number of samples'
      return len(self.graphs)

  def __getitem__(self, index):
      'Generates one sample of data'
      # Select sample

      return self.graphs[index]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphfile", type=str, default='graph_pubmed_2000.json',
                        help="The Json file of graphs ")
    # parser.add_argument("--index", type=str, default='data/pubmed_full/results/cg/faiss_pubmedd_2000.index',
    #                     help="The name for the output embedding file")
                        
    parser.add_argument("--base_dir", type=str, default='data/pubmed_full/results/cg/',
                        help="The dir to save the results and load json file")

                        
    parser.add_argument("--sbertmodel", type=str, default='all-MiniLM-L12-v2',
                        help="https://www.sbert.net/docs/pretrained_models.html")
    parser.add_argument("--faiss-type", type=str, default='flat',
                        help="The type of index, flat or compressed")

    return parser.parse_args()


def main(args):
    REDU_DIM=128
    # DIR = args.indir
    graphs = []
    print('=======LOADING EVENTS JSON=======')
    BASE_DIR = args.base_dir
    ids = []
    
    with open(os.path.join(args.base_dir, args.graphfile)) as ff:
        data = json.load(ff)
        
        for g in tqdm(data, desc="Processing graphs", unit="graph"):
             
            # G = json_graph.node_link_graph(g)
            # graphs.append(graph_to_hashable(G))
            if 'id' in g:
                ids.append(g['id'])     

            graphs.append(' '.join(
                [n['name'].rstrip() for n in g['nodes']]
            ))
    if len(ids) == 0:
        ids = None

    print('=======LOADING MODEL=======')
    BERT_MODEL = args.sbertmodel
    model = SentenceTransformer(BERT_MODEL)


    device = torch.device("cuda")
    # Choose whatever GPU device number you want
    # Make sure to call input = input.to(device) on any input tensors that you feed to the model
    model.to(device)
    # Parameters
    params = {'batch_size': 10000,
            'shuffle': False,
            'num_workers': 1}
    # Generators
    graphs_set = EventsDataset(graphs)
    graphs_generator = torch.utils.data.DataLoader(graphs_set, **params)

    embeddings = []
    print('=======ENCODING..=======')
    for graph_batch in tqdm(graphs_generator):
        # obtain encoded tokens
        with torch.no_grad():
            outputs = model.encode(graph_batch)

        embeddings.append(outputs)
        #dbug
        # break

    embs = np.concatenate(embeddings)

    d = embs.shape[1]
    print('Original dimention:', d)
    print('Reducing dimention to: ',REDU_DIM)
    kernel, bias = compute_kernel_bias(embs)

    kernel = kernel[:, :REDU_DIM]
    
    embs = transform_and_normalize(embs, kernel, bias)

    print('++++++CREATING FAISS INDEX==============')
    #preventing TypeError: in method 'IndexFlat_add', argument 3 of type 'float const *
    if args.faiss_type == 'compressed':
        print('Using compressed faiss index')
        index = create_index_compressed(embs.astype('float32'),     )
        index = faiss.index_gpu_to_cpu(index)
    else :
        print('Using flat faiss index')
        index = create_index_flat(embs.astype('float32'), ids)
    # index_file = os.path.join('out', ONTOLOGY, ONTOLOGY+'.index')
    fnoext = args.graphfile.split('.')[0]

    index_file = os.path.join(BASE_DIR,fnoext+'.index')
    print('Saving index to file')
    faiss.write_index(index, index_file)

    print('Saving kernel and bias')
    kernel_file = os.path.join(BASE_DIR,fnoext+'.kernel')
    bias_file = os.path.join(BASE_DIR,fnoext+'.bias')

    with open(kernel_file,'wb') as handle:
        pickle.dump(kernel,handle)

    with open(bias_file,'wb') as handle:
        pickle.dump(bias,handle)


if __name__ == "__main__":
    main(parse_args())
