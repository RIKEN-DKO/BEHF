import json
from flask import Flask, request, jsonify

from bef.doc_retrieval import VectorSearchDimRedu, FaissIndexGraphEventsDimRedu, MultipleGraphIndexManager
from sentence_transformers import SentenceTransformer
from dkouqe.document_retrieval.multiple_search import (SortByBestScore)
import os
from bef.data.netjson2textaejson import convert_single_to_pubannotation
import pickle
from flask_cors import CORS


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
    from bef.search.bioevent_query_handler import BioEventDataWithDB,BioEventQueryHandler,MultiBioEventQueryHandler
    p = argparse.ArgumentParser()

    p.add_argument("--host", "-H", metavar="ADDRESS", default="0.0.0.0")
    p.add_argument("--port", "-p", default=5555, type=int)
    p.add_argument("--event", "-e", default='cg')
    args = p.parse_args()
    print('Loading query handler...')
    bio_event_data = BioEventDataWithDB(event_type=args.event)
    # seeker  = BioEventQueryHandler(bio_event_data)
    evs=['cg','id']
    seeker = MultiBioEventQueryHandler(evs)
    ################
    
    print("Ready for listening.")
    app.run(host=args.host, port=args.port)
