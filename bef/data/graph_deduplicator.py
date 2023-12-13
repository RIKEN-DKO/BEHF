"""
Event Graph Deduplicator

This script removes duplicate graphs from a collection of graph JSON files. Each graph
is assigned a unique ID, and metadata relating the graphs to document IDs and node spans
are stored in separate JSON files.

Usage:
    python graph_deduplicator.py <input_directory> <output_directory>

Where:
    <input_directory> is a directory containing JSON files with the graph data.
    <output_directory> is a directory where the output JSON files will be stored.

Author: JC Rangel
Date: May 18, 2023
"""

# Here starts your code
import json
import networkx as nx
import argparse
from collections import defaultdict
from tqdm import tqdm
import os
...


import json
import networkx as nx
import argparse
from collections import defaultdict
from tqdm import tqdm
import os

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

def main(args):
    unique_graphs = {}
    graphs_to_docs = defaultdict(list)
    docs_to_graphspans = defaultdict(lambda: defaultdict(list))

    total_graphs = 0
    unique_id = 1  # unique id for each graph
    for filename in tqdm(os.listdir(args.input_dir)):
        if filename.endswith('.json'):
            with open(os.path.join(args.input_dir, filename), 'r') as f:
                graph_dicts = json.load(f)

            for graph_dict in graph_dicts:
                G = create_graph_from_dict(graph_dict)
                graph_hash = graph_to_hashable(G)

                if graph_hash not in unique_graphs:
                    graph_id = len(unique_graphs)
                    graph_dict['id'] = graph_id  # Assign a new unique ID

                    unique_graphs[graph_hash] = graph_dict  # Store the entire graph dictionary
                    unique_id += 1
                else:
                    graph_id = unique_graphs[graph_hash]['id']

                doc_id = graph_dict['graph']['source_doc']
                graphs_to_docs[graph_id].append(doc_id)
                for node in graph_dict['nodes']:
                    #Saving the name instead of the id , because id vary beetwen events
                    docs_to_graphspans[doc_id][graph_id].append({'name': node['name'], 'span': node['span']})

                total_graphs += 1


    unique_graph_dicts = list(unique_graphs.values())  # Use the stored graph dictionaries
    print(f"Total number of unique graphs: {unique_id}")
    print(f"Total number of graphs: {total_graphs}")
    #REmoving unwatd data
    clean_unique_graph = []
    for graph_dict in unique_graph_dicts:
        graph_dict.pop('graph', None)  # Remove the source_doc field
        for node in graph_dict['nodes']:
            node.pop('span', None)  # Remove the span field

        clean_unique_graph.append(graph_dict)


    with open(os.path.join(args.output_dir, 'events_graph.json'), 'w') as f:
        json.dump(clean_unique_graph, f)

    with open(os.path.join(args.output_dir, 'graphs2docs.json'), 'w') as f:
        json.dump(graphs_to_docs, f)

    with open(os.path.join(args.output_dir, 'docs2graphspans.json'), 'w') as f:
        json.dump(docs_to_graphspans, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process graph data.')
    parser.add_argument('input_dir', help='Input directory containing JSON files')
    parser.add_argument('output_dir', help='Output directory for result files')
    
    args = parser.parse_args()

    main(args)
