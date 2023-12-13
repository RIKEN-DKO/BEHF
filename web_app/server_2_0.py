import pickle
import json
from http.server import BaseHTTPRequestHandler

from bef.doc_retrieval import VectorSearchDimRedu, FaissIndexGraphEventsDimRedu, MultipleGraphIndexManager

from sentence_transformers import SentenceTransformer

from dkouqe.document_retrieval.multiple_search import (SortByBestScore)
import os

from bef.data.netjson2textaejson import convert_single_to_pubannotation


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
            result = seeker(
            query = data['query'],
            num_res = data['num_docs'])

            
            # print(result)

            # Singular document.
            if len(result) > 0:
                # return [*result.values()][0]
                return json.dumps( result)
                # return result

            return []

    


    return GetHandler

class BioEventData:
    def __init__(self, event_type='cg', base_dir='../data/pubmed_70s/'):
        self.base_dir = os.path.join(base_dir, 'cg') if event_type == 'cg' else '../data/pubmed_full/results/id/'
        self.emb = None
        self.seeker = None
        self.pmid2data = None
        self.graphs2docs = None
        self.docs2graphspans = None
        self.kernel = None
        self.bias = None
        self.graphs_file = None
        self.index_file = None
        self.load_files_and_initialize()

    def load_files_and_initialize(self):
        BERT_MODEL = 'all-MiniLM-L12-v2'
        self.emb = self.load_transfomer(BERT_MODEL)

        self.pmid2data = self.load_json(os.path.join('..','pmid2data.json'))
        self.graphs2docs = self.load_json('graphs2docs.json')
        self.docs2graphspans = self.load_json('docs2graphspans.json')

        self.kernel = self.load_pickle('events_graph.kernel')
        self.bias = self.load_pickle('events_graph.bias')

        self.graphs_file = os.path.join(self.base_dir, 'events_graph.json')
        self.id2graph = self.load_json('events_graph.json')
        self.index_file = os.path.join(self.base_dir, 'events_graph.index')

        strategy = VectorSearchDimRedu()

        seek_args = {
            'search_type': strategy,
            'index_file': self.index_file,
            'embedding_model': self.emb,
            'kernel': self.kernel,
            'bias': self.bias,
            'id2graph': self.id2graph
        }

        self.seeker = FaissIndexGraphEventsDimRedu(**seek_args)

    def load_json(self, filename):
        with open(os.path.join(self.base_dir, filename), 'r') as handle:
            return json.load(handle)

    def load_pickle(self, filename):
        with open(os.path.join(self.base_dir, filename), 'rb') as handle:
            return pickle.load(handle)

    def load_transfomer(self,BERT_MODEL):
        sentence_emb_model = SentenceTransformer(BERT_MODEL)
        return sentence_emb_model


class BioEventQueryHandler:
    def __init__(self, bio_event_data):
        self.data = bio_event_data

    def map_to_api_format(self, query, seeker_results):
        api = {}
        api['query'] = query
        api['results'] = []
        for i, _ in enumerate(seeker_results['faiss_scores']):

            result = {
                'rank': i+1,
                'total_score': float(seeker_results['faiss_scores'][i]),
            }
            graph_id =  int(seeker_results['records_ids'][i])
            docs_id = self.data.graphs2docs[str(graph_id)]
            docs_title = [self.data.pmid2data[doc_id]['title'] for doc_id in docs_id] 
            docs_abstract = [self.data.pmid2data[doc_id]['abstract'] for doc_id in docs_id] 
            
            
            # graphs_spans = [self.data.docs2graphspans[str(doc_id)][str(graph_id)] for doc_id in docs_id] 
            graph = self.data.id2graph[graph_id]
            #Grab the text of the first document
            nodes_spans = self.data.docs2graphspans[docs_id[0]][str(graph_id)]
            text = docs_title[0] +docs_abstract[0]
            
            #It create the textAE pubannotation.
            #This is dict it need to be converted to JSON by the website
            result['textAE']=convert_single_to_pubannotation(nodes_spans,graph,text)


            result["graph_id"]= graph_id
            documents = []
            for i,doc_id in enumerate(docs_id):

                documents.append({
                        "doc_id" : docs_id[i],
                        "doc_title":docs_title[i],
                        "abstract":docs_abstract[i],
                        # "graph_spans":graphs_spans[i] 
                })

            result['documents'] = documents
            api['results'].append(result)
        return api


    def __call__(self, query, num_res=10):
        return self.map_to_api_format(
            query=query,
            seeker_results=self.data.seeker.retrieve_records(query=query, num_docs=num_res)
        )






if __name__ == "__main__":
    import argparse
    from http.server import HTTPServer
    p = argparse.ArgumentParser()

    p.add_argument("--bind", "-b", metavar="ADDRESS", default="0.0.0.0")
    p.add_argument("--port", "-p", default=5555, type=int)
    p.add_argument("--event", "-e", default='cg')
    args = p.parse_args()
    print('Loading query handler...')
    bio_event_data = BioEventData()
    seeker  = BioEventQueryHandler(bio_event_data)
    ################

    server_address = (args.bind, args.port)
    server = HTTPServer(
        server_address,
        make_handler(seeker),
    )

    try:
        print("Ready for listening.")
        server.serve_forever()
    except KeyboardInterrupt:
        exit(0)
