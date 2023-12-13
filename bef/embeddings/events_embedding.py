#modified from https://github.com/disi-unibo-nlp/ddegk/
import argparse
import os
import shutil

# We must use PyTorch because DDGK needs an old version of TF
os.environ['USE_TORCH'] = 'TRUE'

import bef.data.standoff2graphs_noembs as standoff2graphs_noembs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-prototypes', type=int,
                          required=False, default=16)
    parser.add_argument('--node-embedding-coeff', type=float, required=True, help='The coefficient for the loss related to node embeddings')
    parser.add_argument('--node-label-coeff', type=float, required=True, help='The coefficient for the loss related to node discrete labels')
    parser.add_argument('--edge-label-coeff', type=float, required=True, help='The coefficien or the loss related to edge labels')
    parser.add_argument('--prototype-choice', type=str, required=True)
    parser.add_argument('--num-threads', default=32, type=int)
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to work with")
    parser.add_argument("--scibert-batch-size", type=int, default=2)
    return parser.parse_args()

def main():
    args = parse_args()
    dataset_name = args.dataset
    output_dir = f"results/{dataset_name}"
    args.dataset = f"data/{dataset_name}"
    args.output_file = os.path.join(output_dir, 'graphs.json')
    args.batchsize = args.scibert_batch_size
    args.graphs_file = args.output_file
    args.working_dir = os.path.join(output_dir, 'ddegk')
    print(args)

    if os.path.isdir(args.working_dir):
        shutil.rmtree(args.working_dir)
    os.makedirs(args.working_dir)


    print("\n==== Convert events to graphs no embeddings====")
    standoff2graphs_noembs.main(args)

if __name__ == "__main__":
    main()
