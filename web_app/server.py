import pickle
import json
from http.server import BaseHTTPRequestHandler

from bef.doc_retrieval import VectorSearchDimRedu, FaissIndexGraphEventsDimRedu, MultipleGraphIndexManager

from sentence_transformers import SentenceTransformer

from dkouqe.document_retrieval.multiple_search import (SortByBestScore)
import os


API_DOC = "API_DOC"

"""

"""


def make_handler(doc_retrieval):
    class GetHandler(BaseHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            self.doc_retrieval = doc_retrieval

            super().__init__(*args, **kwargs)

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                bytes(
                    json.dumps(
                        {
                            "schemaVersion": 1,
                            "label": "status",
                            "message": "up",
                            "color": "green",
                        }
                    ),
                    "utf-8",
                )
            )
            return

        def do_HEAD(self):
            # send bad request response code
            self.send_response(400)
            self.end_headers()
            self.wfile.write(bytes(json.dumps([]), "utf-8"))
            return

        def do_POST(self):
            """
            Returns response.

            :return:
            """
            try:
                content_length = int(self.headers["Content-Length"])
                post_data = self.rfile.read(content_length)
                self.send_response(200)
                self.end_headers()

                response = self.generate_response(self.read_json(post_data))

                self.wfile.write(bytes(json.dumps(response), "utf-8"))
            except Exception as e:
                print(f"Encountered exception: {repr(e)}")
                self.send_response(400)
                self.end_headers()
                self.wfile.write(bytes(json.dumps([]), "utf-8"))
            return

        def read_json(self, post_data):
            """
            Reads input JSON message.

            :return: document text and spans.
            """

            data = json.loads(post_data.decode("utf-8"))
            query = data["query"]
            num_docs=None
            if 'num_docs' in data:
                num_docs = data["num_docs"]
            else:
                num_docs = 10

            query = query.replace("&amp;", "&")

            return {'query': query,
            'num_docs': num_docs}

        def generate_response(self, data):
            """
            Generates response for API. Can be either ED only or EL, meaning end-to-end.

            :return: list of tuples for each entity found.
            """

            if len(data['query']) == 0:
                return []

            # Process result.
            # result = self.doc_retrieval.link_entities(text)
            # TODO 10
            result = mseeker.retrieve_records(
            query = data['query'],
            num_docs = data['num_docs'])

            
            # print(result)

            # Singular document.
            if len(result) > 0:
                # return [*result.values()][0]
                return json.dumps(self.map_to_api_format(data['query'], result))
                # return result

            return []

        def map_to_api_format(self, query, server_out):
            """
 
            """
            api = {}
            api['query'] = query

            server_out['graphs'] = server_out['graphs'].tolist()
            # 'docs_ids': array([2193950, 9016379, 2364035]),
            # print(server_out)
            api['results'] = [{
                'rank': i+1,
                'total_score': float(server_out['scores'][i]),
                'document':{
                    "event_id": int(server_out['records_ids'][i]),
                    # "title": server_out['titles'][i],
                    # "abstracts": server_out['abstracts'][i],
                    # "journal_info": server_out['journals'][i],
                    # "year": server_out['years'][i],
                    "nodes" : server_out['graphs'][i]['nodes'],
                    "links" : server_out['graphs'][i]['links'],
                    "pmid" : server_out['graphs'][i]['graph']['source_doc']
                }
            } for i, _ in enumerate(list(server_out['scores']))]

            return api


    return GetHandler


def load_transfomer(BERT_MODEL):
    sentence_emb_model = SentenceTransformer(BERT_MODEL)
    return sentence_emb_model


def load_seekers(type='cg'):
    seekers = {}
    YEAR_RANGE = [2000, 2001]
    if type == 'cg':
        BASE_DIR = '../data/pubmed_full/results/cg/'
    elif type == 'id':
        BASE_DIR = '../data/pubmed_full/results/id/'

    strategy = VectorSearchDimRedu()
    # BERT_MODEL = 'all-mpnet-base-v2'
    BERT_MODEL = 'all-MiniLM-L12-v2'
    sentence_emb_model = load_transfomer(BERT_MODEL)
    # years = ['2000','2001','2002']
    emb = sentence_emb_model

    years = [str(i) for i in range(YEAR_RANGE[0], YEAR_RANGE[1])]
    for year in years:
        graphs_file = os.path.join(BASE_DIR, 'graph_pubmed_' + year + '.json')
        # Check existence

        index_file = os.path.join(BASE_DIR, 'graph_pubmed_' + year + '.index')

        with open(os.path.join(BASE_DIR, 'graph_pubmed_' + year + '.kernel'), 'rb') as handle:
            kernel = pickle.load(handle)

        with open(os.path.join(BASE_DIR, 'graph_pubmed_' + year + '.bias'), 'rb') as handle:
            bias = pickle.load(handle)

        # graph = obonet.read_obo(url)
        seek_args = {'search_type': strategy,
                     'index_file': index_file,
                     'embedding_model': emb,
                     'kernel': kernel,
                     'bias': bias,
                     'graphs_file': graphs_file}

        seekers[year] = FaissIndexGraphEventsDimRedu(**seek_args)

    return seekers


def create_mseeker(event_type='cg'):

    # sentence_emb_model = SentenceTransformer(BERT_MODEL)

    seekers = load_seekers(event_type)

    merge_strategy1 = SortByBestScore()
    merge_strategy_descending = SortByBestScore(ascending=True)
    mseeker = MultipleGraphIndexManager(seekers, merge_strategy1)
    # mseeker._seekers
    # All stuff must loaded from a single function or it will keep loading
    # stuff if you have multiple st.cache functions callling each other
    return mseeker


if __name__ == "__main__":
    import argparse
    from http.server import HTTPServer
    p = argparse.ArgumentParser()

    p.add_argument("--bind", "-b", metavar="ADDRESS", default="0.0.0.0")
    p.add_argument("--port", "-p", default=5555, type=int)
    p.add_argument("--event", "-e", default='cg')
    args = p.parse_args()

    mseeker = create_mseeker(args.event)

    ################

    server_address = (args.bind, args.port)
    server = HTTPServer(
        server_address,
        make_handler(mseeker),
    )

    try:
        print("Ready for listening.")
        server.serve_forever()
    except KeyboardInterrupt:
        exit(0)
