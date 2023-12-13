import argparse
import json
from elasticsearch import Elasticsearch
from bef.index.splade import SpladeBOW

from elasticsearch.helpers import bulk
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("json_file", help="Path to the JSON file")
parser.add_argument("index_name", help="Name of the Elasticsearch index")
args = parser.parse_args()

# Create an instance of SpladeBOW
splade = SpladeBOW()

# Create an Elasticsearch client
# es = Elasticsearch()
es = Elasticsearch(hosts="http://localhost:9200")

print('Loading your data from the JSON file')
with open(args.json_file, "r") as read_file:
    json_data = json.load(read_file)
# python create_elasticsearch_index.py /home/julio/repos/ event_finder/data/pubmed_70s/id/events_graph.json pubmed_70s_id
# Create an Elasticsearch index
index_name = 'data_'+args.index_name
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name)

# Add documents to the index
texts = []
for graph in json_data:
    id = graph['id']
    texts.append(' '.join(node['name'] for node in graph['nodes']))

print('Getting the expanded keywords in SPLADE')
texts_expanded_keywords = splade.get_bows(texts)

def generate_data():
    for i, expanded_keywords in enumerate(texts_expanded_keywords):
        keyws = [keyw[0] for keyw in expanded_keywords]
        document = {
            "_index": index_name,
            "_id": json_data[i]['id'],
            "_source": {
                "name": texts[i],
                "expanded_keywords": keyws,
            }
        }
        yield document

print('Adding it to Elasticsearch')

actions = list(generate_data())
success, _ = bulk(es, actions)

print(f'Successfully indexed {success} documents.')