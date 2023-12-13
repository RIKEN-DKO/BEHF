import json
import networkx as nx
import argparse
from collections import defaultdict
from tqdm import tqdm

def create_graph_from_dict(graph_dict):
    G = nx.DiGraph() if graph_dict['directed'] else nx.Graph()
    
    for node in graph_dict['nodes']:
        G.add_node(node['id'], type=node['type'], name=node['name'])
        
    for link in graph_dict['links']:
        G.add_edge(link['source'], link['target'], key=link['key'])
        
    return G

def graph_to_hashable(G):
    return tuple(sorted(dict(G.degree()).values()))

def main(args):
    # Load the JSON data into a list of dictionaries
    with open(args.input_file) as f:
        graph_dicts = json.load(f)

    unique_graphs = defaultdict(list)
    source_doc_dict = defaultdict(list)

    unique_graph_dicts = []

    for i in tqdm(range(len(graph_dicts)), desc='Processing networks'):
        graph_dict = graph_dicts[i]
        G = create_graph_from_dict(graph_dict)
        graph_hash = graph_to_hashable(G)

        if any(nx.is_isomorphic(G, unique_G, node_match=lambda data1, data2: data1['type'] == data2['type'])
               for unique_G in unique_graphs[graph_hash]):
            continue  

        unique_graphs[graph_hash].append(G)
        graph_dict['id'] = len(unique_graph_dicts)  # Assign a new unique ID
        unique_graph_dicts.append(graph_dict)  # Add the graph dict to the list of unique graph dicts

        source_doc_dict[graph_dict['graph']['source_doc']].append(graph_dict['id'])

    with open(args.unique_networks_file, 'w') as f:
        json.dump(unique_graph_dicts, f)

    with open(args.source_doc_file, 'w') as f:
        json.dump(source_doc_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process graph data.')
    parser.add_argument('input_file', help='Input JSON file containing network data')
    parser.add_argument('unique_networks_file', help='Output JSON file for unique networks')
    parser.add_argument('source_doc_file', help='Output JSON file for source document relations')
    
    args = parser.parse_args()

    main(args)
