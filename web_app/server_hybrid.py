import json
from flask import Flask, request, jsonify

from bef.search.doc_retrieval import VectorSearchDimRedu, FaissIndexGraphEventsDimRedu, MultipleGraphIndexManager
from sentence_transformers import SentenceTransformer
from bef.search.multiple_search import (SortByBestScore)
import os
from bef.data.netjson2textaejson import convert_single_to_pubannotation
import pickle
from flask_cors import CORS
from flask import abort

API_DOC = "API_DOC"


# existing code...
app = Flask(__name__)
CORS(app) # This will enable CORS for all routes

# existing code...

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "schemaVersion": 1,
        "label": "status",
        "message": "up",
        "color": "green",
    }), 200

@app.route('/', methods=['POST'])
def process_request():
    try:
        post_data = request.get_json()
        response = generate_response(read_json(post_data))
        return jsonify(response), 200
    except Exception as e:
        print(f"Encountered exception: {repr(e)}")
        return jsonify([]), 400

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    event_type = data.get('event_type')
    query = data.get('query')
    num_res = data.get('num_res')
    alpha = data.get('alpha', 0.5)  # Set a default value if not provided
    if not event_type or not query or not num_res:
        abort(400, description="Invalid request, 'event_type', 'query', and 'num_res' are required.")
    result = seeker.search(event_type, query, num_res, alpha)
    return jsonify(result), 200


@app.route('/annotations/<event_type>/<doc_id>', methods=['GET'])
def get_doc_annotations(event_type, doc_id):
    result = seeker.get_doc_annotations(event_type, doc_id)
    return jsonify(result), 200

def read_json(data):
    query = data.get("query", "").replace("&amp;", "&")
    num_docs = data.get("num_res")
    events_to_query = data.get("events_to_query")
    type = data.get("type", "search")  # Defaults to 'search' if not provided
    return {'query': query, 'num_docs': num_docs, 'type': type,'events_to_query':events_to_query}

def generate_response(data):
    if data['query'] is None:
        return []
    if len(data['query']) == 0:
        return []
    result = seeker(query = data['query'], num_res = data['num_docs'], type = data['type'],events_to_query=data['events_to_query'])

    if result is None:
        return []
    if len(result) > 0:
        return result

    return []



if __name__ == "__main__":
    import argparse
    import os
    from bef.search.bioevent_query_handler import BioEventDataWithDB,BioEventQueryHandler,MultiBioEventQueryHandler, MultiHybridBioEventQueryHandler

    p = argparse.ArgumentParser()
    p.add_argument("--host", "-H", metavar="ADDRESS", default="0.0.0.0")
    p.add_argument("--data_path", "-D", default="pubmed_70s",help='data/{data_path} ')
    p.add_argument("--port", "-p", default=5555, type=int)
    args = p.parse_args()
    
    # Get a list of all subdirectories of 'data/data_path'
    subdirs = [d for d in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, d))]
    
    print('Loading query handler...')
    seeker = MultiHybridBioEventQueryHandler(subdirs, data_path=args.data_path)
    
    print("Ready for listening.")
    app.run(host=args.host, port=args.port)
