import json
import networkx as nx
import argparse
from collections import defaultdict
import os

# functions to transform graphs into hashable objects
def create_graph_from_dict(graph_dict):
    G = nx.DiGraph() if graph_dict['directed'] else nx.Graph()
    
    for node in graph_dict['nodes']:
        G.add_node(node['id'], type=node['type'], name=node['name'])
        
    for link in graph_dict['links']:
        G.add_edge(link['source'], link['target'], key=link['key'])
        
    return G

def graph_to_hashable(G):
    node_attrs = tuple(sorted((data['name'], data['type']) for node, data in G.nodes(data=True)))
    edge_attrs = tuple(sorted((G[u][v].get('key', '')) for u, v in G.edges()))
    return (len(G.nodes()), len(G.edges()), tuple(sorted(dict(G.degree()).values())), node_attrs, edge_attrs)

# create argument parser
parser = argparse.ArgumentParser()
parser.add_argument("directory", help="Directory where the json files are located")
args = parser.parse_args()

# load graph data
with open(os.path.join(args.directory, 'events_graph.json'), 'r') as f:
    graphs = json.load(f)

# create a dictionary mapping graph ids to their hashable representation
graph_dict = {str(graph['id']): graph_to_hashable(create_graph_from_dict(graph)) for graph in graphs}

# load document-graph relations
with open(os.path.join(args.directory, 'docs2graphspans.json'), 'r') as f:
    doc2graph = json.load(f)

# create a dictionary mapping documents to the set of graphs they contain
doc_graph_sets = {doc_id: {graph_dict[graph_id] for graph_id in doc_graphs.keys()} for doc_id, doc_graphs in doc2graph.items()}

# find the intersection of all graph sets
common_graphs = set.intersection(*doc_graph_sets.values())

# print the result
print(f'The following graphs are common to all documents: {common_graphs}')

# output the common graphs to common.json
with open(os.path.join(args.directory, 'common.json'), 'w') as f:
    json.dump(list(common_graphs), f)
