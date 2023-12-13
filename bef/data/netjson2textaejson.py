"""
This script is used to convert network descriptions in JSON format into TextAE PubAnnotation JSON format. 
The network descriptions represent entity relations in biomedical text.

The input to the script is two JSON files:

1. The first file contains a list of network descriptions in the following format:

[
    {
        "directed": true,
        "multigraph": false,
        "graph": {
            "source_doc": "7860610",
            "dataset": "PUBMED",
            "root": "T28"
        },
        "nodes": [
            ...
        ],
        "links": [
            ...
        ]
    },
    ...
]

2. The second file contains the biomedical text with each text corresponding id `source_doc`, like this:

{
    "7860610": [
        "Sentence 1","sentence 2","sentence n"
    ],
    ...
}

The output of the script is a JSON file in the TextAE PubAnnotation format. 

This script also has an optional feature to crop the text to the span of the nodes. 
This feature can be enabled by passing the '--crop_text' argument in the command line.

Usage:
python script.py networks.json texts.json --crop_text
"""

# The rest of the script follows...

import json
import sys
from typing import Dict, List
import argparse




def get_min_max_span(nodes: List[Dict]) -> (int, int):
    min_span = float('inf')
    max_span = float('-inf')

    for node in nodes:
        min_span = min(min_span, node['span'][0])
        max_span = max(max_span, node['span'][1])

    return min_span, max_span


def convert_to_pubannotation(networks: List[Dict], texts: Dict[str, List[str]], crop_text: bool = False,source_doc = None) -> List[Dict]:
    pubannotations = []

    for network in networks:
        pubannotation = {}
        if "graph" in network:
            source_doc = network["graph"]["source_doc"] 
            pubannotation["sourcedb"] = network["graph"]["dataset"]
            
        pubannotation["sourceid"] = source_doc
        

        # convert nodes to denotations
        denotations = []
        if crop_text:
            min_span, max_span = get_min_max_span(network["nodes"])
            pubannotation["text"] = " ".join(texts[source_doc])[min_span:max_span]
        else:
            pubannotation["text"] = " ".join(texts[source_doc])

        for node in network["nodes"]:
            denotation = {}
            denotation["id"] = node["id"]
            if crop_text:
                denotation["span"] = {"begin": node["span"][0] - min_span, "end": node["span"][1] - min_span}
            else:
                denotation["span"] = {"begin": node["span"][0], "end": node["span"][1]}
            denotation["obj"] = node["type"]
            denotations.append(denotation)
        pubannotation["denotations"] = denotations

        # convert links to relations
        relations = []
        for link in network["links"]:
            relation = {}
            relation["id"] = link["event_id"]
            relation["pred"] = link["key"]
            relation["subj"] = link["source"]
            relation["obj"] = link["target"]
            relations.append(relation)
        pubannotation["relations"] = relations

        pubannotations.append(pubannotation)

    return pubannotations

def convert_networks_to_pubannotation(networks: List[Dict], text: str) -> Dict:
    """
    This function converts a list of network descriptions for a single text into TextAE PubAnnotation JSON format.

    Parameters:

    - networks (List[Dict]): A list of dictionaries representing the networks.
      Each dictionary should have "nodes" and "links" keys.
      The "nodes" key should have a list of dictionaries representing the nodes, where each node dictionary contains "type", "name", "id", and "span" keys.
      The "links" key should have a list of dictionaries representing the links, where each link dictionary contains "key", "event_id", "source", and "target" keys.

    - text (str): The text that corresponds to the networks.

    Returns:

    - Dict: A dictionary in TextAE PubAnnotation JSON format representing the networks.

    Example:

    networks = [
        {
            'directed': True,
            'multigraph': False,
            'nodes': [
                {'type': 'Catabolism', 'name': 'hydrolyse\n', 'id': 'T17', 'span': [192, 201]},
                {'type': 'Simple_chemical', 'name': 'phenylthioacetate\n', 'id': 'T8', 'span': [202, 219]}
            ],
            'links': [
                {'key': 'Theme', 'event_id': 'E1', 'source': 'T17', 'target': 'T8'}
            ],
            'id': 192312
        },
        ...
    ]

    text = "The text corresponding to the networks."

    pubannotation = convert_networks_to_pubannotation(networks, text)
    print(json.dumps(pubannotation, indent=2))
    """
    pubannotation = {}

    pubannotation["text"] = text

    # convert nodes to denotations
    denotations = []
    # print(networks)
    for network in networks:
        for node in network["nodes"]:
            denotation = {}
            denotation["id"] = node["id"]
            denotation["span"] = {"begin": node["span"][0], "end": node["span"][1]}
            denotation["obj"] = node["type"]
            denotations.append(denotation)

    pubannotation["denotations"] = denotations

    # convert links to relations
    relations = []
    for network in networks:
        for link in network["links"]:
            relation = {}
            relation["id"] = link["event_id"]
            relation["pred"] = link["key"]
            relation["subj"] = link["source"]
            relation["obj"] = link["target"]
            relations.append(relation)

    pubannotation["relations"] = relations

    return pubannotation


def convert_single_to_pubannotation(nodes_spans: List[Dict], graph: Dict, text: str, crop_text: bool = False, crop_upto_sent: bool = False) -> Dict:
    """
    This function converts a single network description into TextAE PubAnnotation JSON Format.

    Parameters:

    - nodes_spans (List[Dict]): A list of dictionaries representing nodes in the network.
      Each node dictionary contains a "name" and a "span", where "span" is a list of two integers representing the begin and end of the node in the text.

    - graph (Dict): A dictionary representing the graph of the network.
      The graph contains "nodes" and "links". Each node has a "type", "name", and "id". Each link has a "key", "event_id", "source", and "target".

    - text (str): The text that corresponds to the network.
    
    - crop_text (bool, optional): If set to True, the function will crop the text to the span of the nodes.

    - crop_upto_sent (bool, optional): If set to True and crop_text is also True, the function will include the whole sentence where the min and max span of the nodes are contained.

    Returns:

    - Dict: A dictionary in TextAE PubAnnotation JSON Format representing the network.
    """
    pubannotation = {}

    if crop_text:
        min_span, max_span = get_min_max_span(nodes_spans)
        if crop_upto_sent:
            min_span, max_span = get_sentence_span(text, min_span, max_span)
        pubannotation["text"] = text[min_span:max_span]
    else:
        pubannotation["text"] = text

    # convert nodes to denotations
    denotations = []
    for node in nodes_spans:
        denotation = {}
        denotation["id"] = next((g_node["id"] for g_node in graph["nodes"] if g_node["name"].strip() == node["name"].strip()), None)
        if crop_text:
            if crop_upto_sent:
                node_span_begin = node["span"][0] - min_span
                node_span_end = node["span"][1] - min_span
            else:
                node_span_begin = node["span"][0] - min_span
                node_span_end = node["span"][1] - min_span
        else:
            node_span_begin = node["span"][0]
            node_span_end = node["span"][1]
        denotation["span"] = {"begin": node_span_begin, "end": node_span_end}
        denotation["obj"] = next((g_node["type"] for g_node in graph["nodes"] if g_node["name"].strip() == node["name"].strip()), None)
        if denotation["id"] and denotation["obj"]:
            denotations.append(denotation)

    pubannotation["denotations"] = denotations

    # convert links to relations
    relations = []
    for link in graph["links"]:
        relation = {}
        relation["id"] = link["event_id"]
        relation["pred"] = link["key"]
        relation["subj"] = link["source"]
        relation["obj"] = link["target"]
        relations.append(relation)

    pubannotation["relations"] = relations

    return pubannotation

def get_sentence_span(text: str, min_span: int, max_span: int) -> (int, int):
    # Find the start of the sentence
    start = text.rfind('.', 0, min_span) + 1

    # Find the end of the sentence
    end = text.find('.', max_span)
    if end == -1:
        end = len(text)

    return start, end



def main() -> None:
    parser = argparse.ArgumentParser(description='Convert networks descriptions into TextAE PubAnnotation JSON Format.')
    parser.add_argument('networks_filename', help='Input JSON file with network descriptions')
    parser.add_argument('texts_filename', help='Input JSON file with biomedical texts')
    parser.add_argument('--crop_text', action='store_true', help='Crop the text to the span of the nodes')

    args = parser.parse_args()

    with open(args.networks_filename, 'r') as f:
        networks = json.load(f)

    with open(args.texts_filename, 'r') as f:
        texts = json.load(f)

    pubannotations = convert_to_pubannotation(networks, texts, args.crop_text)

    with open('output.json', 'w') as f:
        json.dump(pubannotations, f, indent=2)

if __name__ == "__main__":
    main()
