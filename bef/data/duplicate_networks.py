"""
Find all the isomorphics graphs in the input file. And output the with no duplicates
"""
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


def are_graphs_isomorphic(graph_dict1, graph_dict2):
    G1 = create_graph_from_dict(graph_dict1)
    G2 = create_graph_from_dict(graph_dict2)

    return nx.is_isomorphic(G1, G2, node_match=lambda data1, data2: data1['type'] == data2['type'])


def main(args):
    # Load the JSON data into a list of dictionaries
    with open(args.input_file) as f:
        graph_dicts = json.load(f)

    # Detect and remove duplicates
    unique_graph_dicts = []
    source_doc_dict = defaultdict(list)

    # Use tqdm to add a progress bar
    for i in tqdm(range(len(graph_dicts)), desc='Processing networks'):
        graph_dict = graph_dicts[i]

        # Check if the current graph is isomorphic to any of the unique graphs
        if any(are_graphs_isomorphic(graph_dict, unique_graph) for unique_graph in unique_graph_dicts):
            continue  # Skip this graph if it's a duplicate

        # Otherwise, add it to the list of unique graphs
        unique_graph_dicts.append(graph_dict)

        # Assign a new unique ID to the graph
        graph_dict['id'] = i

        # Update the source_doc dictionary
        source_doc_dict[graph_dict['graph']['source_doc']].append(i)

    # Save the modified list of networks to a new JSON file
    with open(args.unique_networks_file, 'w') as f:
        json.dump(unique_graph_dicts, f)

    # Save the source_doc dictionary to a new JSON file
    with open(args.source_doc_file, 'w') as f:
        json.dump(source_doc_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process graph data.')
    parser.add_argument(
        'input_file', help='Input JSON file containing network data')
    parser.add_argument('unique_networks_file',
                        help='Output JSON file for unique networks')
    parser.add_argument('source_doc_file',
                        help='Output JSON file for source document relations')

    args = parser.parse_args()

    main(args)
