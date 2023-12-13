
import warnings
#ignoring sklearn deprecation warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn


import numpy as np
import json
import torch
import argparse
from sentence_transformers import SentenceTransformer





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
    parser.add_argument("--indir", type=str, default='/home/julio/repos/ddegk/results/pubmed_small_4/',
                        help="The directory where DeepEventMine graphs.json exists ")
    parser.add_argument("--outdir", type=str, default='/home/julio/repos/event_finder/data/pubmed_small_graph/',
                        help="The directory where embeddings.json will be create")
    parser.add_argument("--outname", type=str, default='sentence_bert_embs_grap.json',
                        help="The name for the output embedding file")
                        
    parser.add_argument("--sbertmodel", type=str, default='all-mpnet-base-v2',
                        help="https://www.sbert.net/docs/pretrained_models.html")

    return parser.parse_args()


def main(args):

    DIR = args.indir
    graphs = []
    print('=======LOADING EVENTS JSON=======')
    with open(DIR + 'graphs.json') as ff:
        for g in json.load(ff):
            graphs.append(' '.join(
                [n['name'].rstrip() for n in g['nodes']]
            ))


    print('=======LOADING MODEL=======')
    BERT_MODEL = args.sbertmodel
    model = SentenceTransformer(BERT_MODEL)


    device = torch.device("cuda")
    # Choose whatever GPU device number you want
    # Make sure to call input = input.to(device) on any input tensors that you feed to the model
    model.to(device)
    # Parameters
    params = {'batch_size': 512,
            'shuffle': False,
            'num_workers': 1}
    # Generators
    graphs_set = EventsDataset(graphs)
    graphs_generator = torch.utils.data.DataLoader(graphs_set, **params)

    embeddings = []
    print('=======ENCODING..=======')
    for graph_batch in graphs_generator:
        # obtain encoded tokens
        with torch.no_grad():
            outputs = model.encode(graph_batch)

        embeddings.append(outputs)

    embs = np.concatenate(embeddings).tolist()

    data = []
    for i in range(len(embs)):
        data.append({
            'idx': i,
            'embedding': embs[i]
        })

    with open(args.outdir + args.outname, 'w') as outfile:
        json.dump(data, outfile)


if __name__ == "__main__":
    main(parse_args())
