from bef.search.multiple_search import (MultipleRetrievalManager, MergeStrategy,
                                                       MultipleOntologiesManager, SortByBestScore)


from bef.search.semantic_search import FaissIndex, VectorSearch, SearchContext, SearchType
import os
from sentence_transformers import SentenceTransformer

import json
from bef.search.search_utils import (vector_search,
                              search_and_print, search_apply_kernel,)
from bef.search.vector_utils import transform_and_normalize


import faiss
import pickle
import numpy as np
from typing import Dict


class VectorSearchDimRedu(SearchType):

    def retrieve_records(self, query, model, index, kernel, bias, num_results):
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
        vector = model.encode([query])

        vector = transform_and_normalize(vector, kernel, bias)
        distances, ids = index.search(
            np.array(vector).astype("float32"), k=num_results)

        return distances.flatten(), ids.flatten()


class FaissIndexGraphEventsDimRedu(SearchContext):
    """Contains and faiss index and the ontology 
    """
    #TODO use dataclass

    def __init__(self,
                 search_type,
                 id2graph,
                 index_file,
                 embedding_model,
                 kernel,
                 bias,
                 convert_to_scores= True
                 ) -> None:

        self._search_type = search_type
        self._embedding_model = embedding_model
        self.kernel = kernel
        self.bias = bias
        self._is_active = True
        self.convert_to_scores = convert_to_scores
        print('Loading graph ...')
        self.graphs = id2graph
        #loading with no embeddings
        # with open(DATA_DIR + 'graphs.json') as ff:
        # with open(graphs_file) as ff:
        #     for g in json.load(ff):
        #         # for node in g['nodes']:
        #         #     node.pop('embedding')
        #         self.graphs.append(g)

        self.index = faiss.read_index(index_file)

    def retrieve_records(self, query: str, num_docs: int):
        distances, ids = self._search_type.retrieve_records(query,
                                                            self._embedding_model,
                                                            self.index,
                                                            self.kernel,
                                                            self.bias,
                                                            num_docs)

        results = {}
        if self.convert_to_scores:
            scores = [1.0 / (1+dist) for dist in distances]

            # Create pairs of (score, docid), sort them by score in descending order,
            # and then split the sorted pairs back into separate lists of scores and docids.
            sorted_pairs = sorted(zip(scores, ids), key=lambda pair: pair[0], reverse=True)
            sorted_scores, sorted_docids = zip(*sorted_pairs)
            results['faiss_scores'] = sorted_scores
            results['records_ids'] = sorted_docids

        else:
            results['faiss_scores'] = distances
            results['records_ids'] = ids
        # results['nodes'] =
        return results

    def search_and_print_ontology(self, query: str, num_docs: int, sentence_emb_model, kernel, bias) -> None:
        """Search and print the ontology, for debuging.

        Args:
            query (str): [description]
            num_docs (int): [description]
        """
        search_apply_kernel(query, self._embedding_model,
                            self.index, self.graphs, kernel, bias)


class MultipleGraphIndexManager(MultipleOntologiesManager):
    """Handles when we want to retrieve information from multiple index
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._results = self.empty_results()

    def retrieve_records(self, query: str, num_docs: int) -> Dict:
        # results = self._results
        #TODO numdocs should be specific for each seeker?
        self._results = self.empty_results()
        for name, seeker in self._seekers.items():
            if seeker._is_active:
                new_results = seeker.retrieve_records(query, num_docs)
                if 'faiss_scores' in new_results:

                    self._results['scores'] = np.concatenate(
                        [self._results['scores'], new_results['faiss_scores']])
                    self._results['records_ids'] = np.concatenate(
                        [self._results['records_ids'],
                         new_results['records_ids']])

                    new_refs = [seeker] * len(new_results['faiss_scores'])
                    self._results['seeker_ref'] += new_refs

                    graphs = seeker.graphs
                    nodes,sourcedoc_id,links = [],[],[]
                    for id_ in new_results['records_ids']:
                        # node = graphs[id_]['nodes']
                        
                        nodes.append(graphs[id_])
                        # sourcedoc_id.append(graphs[id_]['graph']['source_id'])
                        # links.append(graphs[id_]['links'])


                    # self._results['nodes'] = np.concatenate(
                    #     [self._results['nodes'], nodes])
                    self._results['graphs'] += nodes
                    # self._results['sourcedoc_id'] += sourcedoc_id
                    # self._results['link'] += links

        self._results = self.merge_strategy.merge_records(self._results, query)
        return self._results
