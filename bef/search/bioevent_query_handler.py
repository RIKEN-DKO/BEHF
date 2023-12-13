import json
from typing import Dict


from bef.search.doc_retrieval import VectorSearchDimRedu, FaissIndexGraphEventsDimRedu
from sentence_transformers import SentenceTransformer

import os
from bef.data.netjson2textaejson import convert_single_to_pubannotation,convert_networks_to_pubannotation
import pickle

def load_transfomer(BERT_MODEL):
    sentence_emb_model = SentenceTransformer(BERT_MODEL)
    return sentence_emb_model

class BioEventData:
    """
    A class used to represent biological event data.

    This class provides methods to load necessary files (such as JSON and Pickle files) 
    and initialize data needed for bio-event processing. 

    Attributes
    ----------
    base_dir : str
        The base directory to load the data files.
    emb : SentenceTransformer
        Transformer model used for sentence embeddings.
    seeker : FaissIndexGraphEventsDimRedu
        Index searcher for graph events with dimension reduction.
    pmid2text : dict
        A dictionary that maps PubMed IDs to corresponding texts.
    graphs2docs : dict
        A dictionary that maps graph IDs to corresponding documents.
    docs2graphspans : dict
        A dictionary that maps document IDs to corresponding graph spans.
    kernel : object
        Kernel loaded from a Pickle file.
    bias : object
        Bias loaded from a Pickle file.
    graphs_file : str
        Path to the 'events_graph.json' file.
    index_file : str
        Path to the 'events_graph.index' file.

    Methods
    -------
    load_files_and_initialize()
        Load necessary files and initialize data.
    load_json(filename)
        Load a JSON file.
    load_pickle(filename)
        Load a Pickle file.
    load_transfomer(BERT_MODEL)
        Load a transformer model.
    """

    def __init__(self, event_type='cg', base_dir='../data/pubmed_70s/'):
        self.base_dir = os.path.join(base_dir, event_type)
        self.emb = None
        self.seeker = None
        self.pmid2text = None
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

        self.pmid2text = self.load_json(os.path.join('..','pmid2text.json'))
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

from pymongo import MongoClient

class MongoDBDictWrapper:
    def __init__(self, collection):
        self.collection = collection

    def __getitem__(self, key):
        result = self.collection.find_one({"_id": key})
        if result:
            return result['value']
        else:
            raise KeyError(f"No record found with _id: {key}")

    def __contains__(self, key):
        return self.collection.find_one({"_id": key}) is not None


class BioEventDataWithDB(BioEventData):
    """
    A class used to represent biological event data, with a MongoDB database.

    This class inherits from BioEventData, but replaces the 'pmid2text' dictionary 
    with a MongoDB collection, providing a more efficient way to handle large datasets.

    The behavior of accessing 'pmid2text' remains the same: pmid2text['ID'] will query the MongoDB collection for a document with the given ID.

    Attributes
    ----------
    pmid2text : MongoDBDictWrapper
        A wrapper for a MongoDB collection that behaves like a dictionary.

    Methods
    -------
    load_files_and_initialize()
        Load necessary files (excluding the JSON file used in the parent class) 
        and initialize data.
    """
    def __init__(self,embedding_model, event_type='cg', base_dir='../data/pubmed_70s/', db_uri='mongodb://localhost:27017/'):
        self.base_dir = os.path.join(base_dir, event_type) 
        self.emb = embedding_model
        self.seeker = None
        self.pmid2text = None
        self.graphs2docs = None
        self.docs2graphspans = None
        self.kernel = None
        self.bias = None
        self.graphs_file = None
        self.index_file = None

        # Initialize a MongoDB client
        client = MongoClient(db_uri)

        # Access the database and collection
        db = client['pmid2text']
        collection = db['pubMED']

        # Replace the dictionary with the MongoDBDictWrapper instance
        self.pmid2text = MongoDBDictWrapper(collection)

        self.load_files_and_initialize()

    def load_files_and_initialize(self):
        #This should not be loading
        # BERT_MODEL = 'all-MiniLM-L12-v2'
        # self.emb = self.load_transfomer(BERT_MODEL)

        self.graphs2docs = self.load_json('graphs2docs.json')
        self.docs2graphspans = self.load_json('docs2graphspans.json')

        self.kernel = self.load_pickle('events_graph.kernel')
        self.bias = self.load_pickle('events_graph.bias')

        self.graphs_file = os.path.join(self.base_dir, 'events_graph.json')
        self.id2graph = self.load_json('events_graph.json')
        self.index_file = os.path.join(self.base_dir, 'events_graph.index')

        strategy = VectorSearchDimRedu()

        self.faiss_seek_args = {
            'search_type': strategy,
            'index_file': self.index_file,
            'embedding_model': self.emb,
            'kernel': self.kernel,
            'bias': self.bias,
            'id2graph': self.id2graph
        }

        # self.seeker = FaissIndexGraphEventsDimRedu(**seek_args)


class BioEventDataFullDB(BioEventData):
    """
    A class used to represent biological event data, with a MongoDB database.

    This class inherits from BioEventData, but replaces all data 
    with a MongoDB collection, providing a more efficient way to handle large datasets.

    The behavior of accessing 'pmid2text' remains the same: pmid2text['ID'] will query the MongoDB collection for a document with the given ID.

    Attributes
    ----------
    pmid2text : MongoDBDictWrapper
        A wrapper for a MongoDB collection that behaves like a dictionary.

    """
    def __init__(self,embedding_model, event_type='cg', base_dir='../data/pubmed_70s/', db_uri='mongodb://localhost:27017/'):
        self.base_dir = os.path.join(base_dir, event_type) 
        self.emb = embedding_model
        self.seeker = None
        self.pmid2text = None
        self.graphs2docs = None
        self.docs2graphspans = None
        self.kernel = None
        self.bias = None
        self.graphs_file = None
        self.index_file = None

        # Initialize a MongoDB client
        client = MongoClient(db_uri)

        # Access the database and collection
        db = client['pmid2text']
        collection = db['pubMED']
        self.pmid2text = MongoDBDictWrapper(collection)


        db = client[event_type+'_'+'graphs2docs']
        collection = db['data']
        self.graphs2docs = MongoDBDictWrapper(collection)

        db = client[event_type+'_'+'docs2graphspans']
        collection = db['data']
        self.docs2graphspans = MongoDBDictWrapper(collection)

        self.kernel = self.load_pickle('events_graph.kernel')
        self.bias = self.load_pickle('events_graph.bias')

        self.graphs_file = os.path.join(self.base_dir, 'events_graph.json')

        db = client[event_type+'_'+'events_graph']
        collection = db['data']
        self.id2graph = MongoDBDictWrapper(collection)  

        self.index_file = os.path.join(self.base_dir, 'events_graph.index')

        strategy = VectorSearchDimRedu()

        self.faiss_seek_args = {
            'search_type': strategy,
            'index_file': self.index_file,
            'embedding_model': self.emb,
            'kernel': self.kernel,
            'bias': self.bias,
            'id2graph': self.id2graph
        }




        # self.seeker = FaissIndexGraphEventsDimRedu(**seek_args)


class BioEventQueryHandler:
    def __init__(self, bio_event_data,seeker,scores_name='faiss_scores'):
        self.data = bio_event_data
        self.seeker = seeker
        self.score_name = scores_name

    def map_to_api_format(self, query, seeker_results):
        api = {}
        api['query'] = query
        api['results'] = []
        for i, _ in enumerate(seeker_results[self.score_name]):

            result = {
                'rank': i+1,
                'total_score': float(seeker_results[self.score_name][i]),
            }
            graph_id =  int(seeker_results['records_ids'][i])
            docs_id = self.data.graphs2docs[str(graph_id)]
            docs_text = [self.data.pmid2text[doc_id] for doc_id in docs_id] 

            
            
            # graphs_spans = [self.data.docs2graphspans[str(doc_id)][str(graph_id)] for doc_id in docs_id] 
            graph = self.data.id2graph[graph_id]
            #Grab the text of the first document
            nodes_spans = self.data.docs2graphspans[docs_id[0]][str(graph_id)]
            text = docs_text[0]
            
            #It create the textAE pubannotation.
            #This is dict it need to be converted to JSON by the website
            result['textAE']=validate_textAE(convert_single_to_pubannotation(nodes_spans,graph,text,crop_text=True,crop_upto_sent=True))


            result["graph_id"]= graph_id
            documents = []
            for i,doc_id in enumerate(docs_id):

                documents.append({
                        "doc_id" : docs_id[i],
                        "doc_text": docs_text[i]
                        # "graph_spans":graphs_spans[i] 
                })

            result['documents'] = documents
            api['results'].append(result)
        return api

    def get_networks_wspans(self, network_spans: Dict):

        full_nets = []
        for net_id,nodes in network_spans.items():
            net = self.data.id2graph[int(net_id)]
            # print('net:',net)
            for node in nodes:
                # print(node['name'], '==?', net['nodes'][i]['name'])
                for net_node in net['nodes']:
                    if node['name'] == net_node['name']:
                        # print('enter')
                        net_node['span'] = node['span']

            full_nets.append(net)

        return full_nets
            
    def get_doc_annotations(self, doc_id):
        if doc_id not in self.data.docs2graphspans:
            print(f"No annotations found for document with id: {doc_id}")
            return None 
        # print('before:',self.data.docs2graphspans[doc_id])
        full_nets = self.get_networks_wspans(self.data.docs2graphspans[doc_id])
        # print('after',full_nets)
        textAE = convert_networks_to_pubannotation(full_nets, self.data.pmid2text[doc_id])

        # Validate the textAE before returning it
        return validate_textAE(textAE)

    
    def __call__(self, query, num_res, type="search"):
        if type == "search":
            return self.map_to_api_format(
                query=query,
                seeker_results=self.seeker.retrieve_records(query=query, num_docs=num_res)
            )
        elif type == "get_doc_annotations":
            return self.get_doc_annotations(query)
        else:
            raise ValueError(f"Invalid type: {type}. Expected 'search' or 'get_doc_annotations'.")
        
class BioEventQueryHandlerEL(BioEventQueryHandler):
    def __init__(self, bio_event_data, seeker, scores_name='faiss_scores'):
        super().__init__(bio_event_data, seeker, scores_name)

    def __call__(self, query, type="search"):
        if type == "search":
            return self.map_to_api_format(
                query=query,
                seeker_results=self.seeker.retrieve_records(query=query)
            )
        elif type == "get_doc_annotations":
            return self.get_doc_annotations(query)
        else:
            raise ValueError(f"Invalid type: {type}. Expected 'search' or 'get_doc_annotations'.")

    

from heapq import merge

class MultiBioEventQueryHandler:
    def __init__(self, event_types):
        self.event_handlers = {event_type: BioEventQueryHandler(BioEventDataWithDB(event_type=event_type))
                               for event_type in event_types}

    def __call__(self, query, num_res, type="search", events_to_query=None):
        if events_to_query is None:
            events_to_query = self.event_handlers.keys()

        elif type == "get_doc_annotations":
            textAEs = []
            for event_type in events_to_query:
                if event_type not in self.event_handlers:
                    raise ValueError(f"Unknown event type: {event_type}")
                result = self.event_handlers[event_type](query, num_res, type)
                if result is not None:  # only append result if it's not None
                    textAEs.append(result)

            return self.merge_textAEs(textAEs) if textAEs else None  # return None if no annotations were found
        
        elif type == "search":
            results = []
            for event_type in events_to_query:
                if event_type not in self.event_handlers:
                    raise ValueError(f"Unknown event type: {event_type}")
                result = self.event_handlers[event_type](query, num_res, type)
                results.append(result['results'])

            merged_results = list(merge(*results, key=lambda x: x['total_score']))
            return {
                'query': query,
                'results': merged_results
            }
        else:
            raise ValueError(f"Invalid type: {type}. Expected 'search' or 'get_doc_annotations'.")

    @staticmethod
    def merge_textAEs(textAEs):
        if len(textAEs) == 1:
            return textAEs[0]

        merged = textAEs[0].copy()
        id_counter = max(int(d['id'][1:]) for d in merged['denotations'])
        relation_id_counter = max(int(r['id'][1:]) for r in merged['relations'])

        for textAE in textAEs[1:]:
            for denotation in textAE['denotations']:
                id_counter += 1
                denotation['id'] = 'T' + str(id_counter)
                merged['denotations'].append(denotation)
                
            for relation in textAE['relations']:
                relation_id_counter += 1
                relation['id'] = 'E' + str(relation_id_counter)
                relation['subj'] = 'T' + str(int(relation['subj'][1:]) + id_counter)
                relation['obj'] = 'T' + str(int(relation['obj'][1:]) + id_counter)
                merged['relations'].append(relation)

        return validate_textAE(merged)

from bef.search.elasticsearch import ElasticSearchSeeker
from bef.search.hybrid import HybridSeeker

class MultiHybridBioEventQueryHandler(MultiBioEventQueryHandler):
    """
    Handles multiple types of hybrid bio-event queries. That is, it can query an speficit 
    hybrid seeker.  

    This class extends the functionality of MultiBioEventQueryHandler to handle hybrid bio-event seekers. 
    It initializes the appropriate FAISS and Elasticsearch seekers based on the provided event types and 
    encapsulates them in a HybridSeeker. Each HybridSeeker can then handle queries for its specific event type.

    Attributes:
        event_handlers (dict): A dictionary mapping event types to their corresponding HybridSeeker.

    Args:
        event_types (list): List of event types for which a HybridSeeker should be initialized.

    Usage : 
    mhybrid_seeker =MultiHybridBioEventQueryHandler(['id','cg'])
    mhybrid_seeker.search(event_type='id',query='melanoma treatment malignant',num_res=20)
    mhybrid_seeker.get_doc_annotations(event_type='cg',doc_id='4460956')
    """
    def __init__(self, event_types,data_path = '../data/pubmed_70s'):
        self.event_handlers = {}
        data_name = data_path.split('/')[-1]
        #TODO there must be a config file inside data_path 
        BERT_MODEL = 'all-MiniLM-L12-v2'
        embedding_model = load_transfomer(BERT_MODEL)
        for event_type in event_types:
            bio_event_data = BioEventDataFullDB(embedding_model=embedding_model,event_type=event_type,base_dir=data_path)

            faiss_seeker = FaissIndexGraphEventsDimRedu(**bio_event_data.faiss_seek_args)
            seeker  = BioEventQueryHandler(bio_event_data,faiss_seeker)
            index_name = f'data_{data_name}_{event_type}'  
            print('EL index:',index_name)
            elastic_seeker = ElasticSearchSeeker(index_name=index_name)
            el_seeker = BioEventQueryHandlerEL(bio_event_data,elastic_seeker,scores_name='scores')
            hybrid_seeker = HybridSeeker(seeker,el_seeker)

            self.event_handlers[event_type] = hybrid_seeker

    def get_doc_annotations(self, event_type, doc_id):
        """
        Retrieves document TextAE annotations for a specific event type.

        Args:
            event_type (str): The event type for which document annotations should be retrieved.
            doc_id (str): The ID of the document to retrieve annotations for.

        Returns:
            dict or None: A dictionary containing document annotations if found, else None.

        Raises:
            ValueError: If an unknown event type is provided.
        """
        return self.event_handlers[event_type].get_doc_annotations(doc_id)

    def search(self, event_type, query, num_res,alpha=0.1):
        """
        Performs a search query for a specific event type.

        Args:
            event_type (str): The event type for which the search should be performed.
            query (str): The query string.
            num_res (int): The number of search results to retrieve.

        Returns:
            dict: A dictionary containing the search results.

        Raises:
            ValueError: If an unknown event type is provided.
        """
        if event_type not in self.event_handlers:
            raise ValueError(f"Unknown event type: {event_type}")
        return self.event_handlers[event_type].search(query=query, num_res=num_res,alpha=alpha)




def validate_textAE(textAE):
    # A set to store unique ids
    unique_ids = set()

    # Iterate over denotations and blocks
    for category in ['denotations', 'blocks']:
        if category in textAE:
            valid_items = []
            for item in textAE[category]:
                if item['id'] not in unique_ids:
                    unique_ids.add(item['id'])
                    valid_items.append(item)

            # Replace the original list with the validated list
            textAE[category] = valid_items

    # Check and validate relations
    if 'relations' in textAE:
        valid_relations = []
        for relation in textAE['relations']:
            # Check if the entities exist in the unique_ids set
            if relation['subj'] in unique_ids and relation['obj'] in unique_ids:
                valid_relations.append(relation)

        # Replace the original relations with the validated relations
        textAE['relations'] = valid_relations

    return textAE
