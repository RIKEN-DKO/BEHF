# import sys
# sys.path.insert(0, "..")

from abc import abstractmethod
from typing import Dict
import faiss
import numpy as np
import time
import pprint
import obonet
from bef.index.faiss_indexes import create_index_flat
import re

class SearchType:
    @abstractmethod
    def retrieve_records(self) :
        pass

#context they acts as con
class SearchContext:

    def __init__(self, search_type: SearchType) -> None:
        """
        SearchContext will acts as containers of data for each types of searching.
        """

        self._search_type = search_type

    @property
    def search_type(self) -> SearchType:
        """
        The Context maintains a reference to one of the SearchType objects. The
        Context does not know the concrete class of a SearchType. It should work
        with all strategies via the SearchType interface.
        """

        return self._search_type
    
    @search_type.setter
    def search_type(self,search_type:SearchType) -> None:
        """
        The Context maintains a reference to one of the SearchType objects. The
        Context does not know the concrete class of a SearchType. It should work
        with all strategies via the SearchType interface.
        """
        self._search_type = search_type


class FaissIndex(SearchContext):
    def __init__(self, 
                search_type: SearchType, 
                index_file: str, 
                ids_index_file: str,
                embedding_model,
                name='ontology',
                 *args, **kwargs) -> None:

        #npy case, we need to create the index in place
        if index_file.split('/')[-1].split('.')[-1] == 'npy':
            # print('Loading Numpy embeddings into Faiss index..')
            # np.load(index_file)
            self._index,onto_ids = create_index_flat(index_file,ids_index_file,return_ids=True)

        else:
            print('Loading index ...')
            self._index = faiss.read_index(index_file)
        #TODO double loading of ids_index_file from create index_flat

            onto_ids = np.load(ids_index_file)
        self._index2ontoid = {i: ontoid for i, ontoid in enumerate(onto_ids)}
        
        self._search_type = search_type
        self._embedding_model = embedding_model

        print('Finished: Loading index')
        #This a flag to be used byt eh multiple search to filter if the search 
        #should be done on this context
        self._is_active = True
        # TODO add say my name of the context
        self._name = name

    def retrieve_records(self, query: str, num_docs: int):
        distances,ids = self._search_type.retrieve_records([query], 
        self._embedding_model, 
        self._index, num_docs,10)

        results = {}
        results['faiss_scores'] = distances
        results['records_ids'] = np.array([self._index2ontoid[id] for id in ids])
        return results
        
    def say_my_name():
        return self._name

    
def print_ontology(ids,distances,graph,index2id) -> None:

    for i in range(len(ids) - 1, 0, -1):
        id_ = ids[i]
        dist = distances[i]
        # print('id: ', id_, 'distance: ', dist)
        # pprint.pprint(graph.nodes[index2id[id_]])
        # print('')
        print_node_ontology(id_,dist,graph,index2id)


def print_node_ontology(id, score, graph, index2id) -> None:

    print('id: ', id, 'score: ', score)
    pprint.pprint(graph.nodes[index2id[id]])
    print('')


class FaissIndexAndObonetOntology(FaissIndex):
    """Contains and faiss index and the ontology 
    """

    def __init__(self, ontology_file, *args, **kwargs
                 ) -> None:
        
        super().__init__(*args, **kwargs)
        print('Loading ontology ...')
        self._ontology_graph = obonet.read_obo(ontology_file)

        self._id2name = {id_: data.get(
            'name') for id_, data in self._ontology_graph.nodes(data=True)}

        print('Finished loading ontology')

    def search_and_print_ontology(self, query: str, num_docs: int) -> None:
        """Search and print the ontology, for debuging.

        Args:
            query (str): [description]
            num_docs (int): [description]
        """
        start = time.time()
        distances, ids = self._search_type.retrieve_records([query],
                                                            self._embedding_model,
                                                            self._index, num_docs, 10)
        distances = distances.flatten().tolist()
        ids = ids.flatten().tolist()
        print('Searching took: ', time.time()-start)

        print_ontology(ids,distances,self._ontology_graph,self._index2ontoid)




## Search types        

class VectorSearch(SearchType):

    def retrieve_records(self,query, model, index, num_results=10, nprobe=1):
        """Tranforms query to vector using a pretrained, sentence-level 
        DistilBERT model and finds similar vectors using FAISS.
        Args:
            query (str): User query that should be more than a sentence long.
            model (sentence_transformers.SentenceTransformer.SentenceTransformer)
            index (`numpy.ndarray`): FAISS index that needs to be deserialized.
            num_results (int): Number of results to return.
        Returns:
            D (:obj:`numpy.array` of `float`): Distance between results and query.
            I (:obj:`numpy.array` of `int`): Paper ID of the results.
        
        """
        vector = model.encode(list(query))
        # index.nprobe = nprobe
        D, I = index.search(np.array(vector).astype("float32"), k=num_results)
        return D.flatten(), I.flatten()

#strategy = FaissIndexSearch()
#ncbisearch = FaissIndex(strategy,index_file,ids_index_file)
#ncbisearch.



