import argparse
import json
from elasticsearch import Elasticsearch
from bef.index.splade import SpladeBOW
from tqdm import tqdm
from elasticsearch.helpers import bulk

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("json_file", help="Path to the JSON file")
parser.add_argument("index_name", help="Name of the Elasticsearch index")
parser.add_argument("--no_splade", action="store_true", help="If set, prevent executing 'splade.get_bows(texts)'")
args = parser.parse_args()

# Create an instance of SpladeBOW
splade = SpladeBOW()

# Create an Elasticsearch client
# es = Elasticsearch()
es = Elasticsearch(hosts="http://localhost:9200")

print('Loading your data from the JSON file')
with open(args.json_file, "r") as read_file:
    json_data = json.load(read_file)

# Create an Elasticsearch index
index_name = 'data_'+args.index_name
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name)

# Add documents to the index
texts = []
for graph in json_data:
    id = graph['id']
    texts.append(' '.join(node['name'] for node in graph['nodes']))

if args.no_splade:
    print('Skipping the expanded keywords in SPLADE')
    texts_expanded_keywords = []
else:
    print('Getting the expanded keywords in SPLADE')
    texts_expanded_keywords = splade.get_bows(texts)

# Prepare actions for bulk indexing
actions = []
print('Preparing documents for indexing')
for i, text in enumerate(tqdm(texts, desc="Processing texts")):
    if not args.no_splade:
        keyws = [keyw[0] for keyw in texts_expanded_keywords[i]]
    else:
        keyws = []
    action = {
        "_index": index_name,
        "_id": json_data[i]['id'],
        "_source": {
            "name": text,
            "expanded_keywords": keyws,
        }
    }
    actions.append(action)

# Perform bulk indexing
print('Adding it to Elasticsearch')
success, _ = bulk(es, actions)
print(f'Successfully indexed {success} documents')
