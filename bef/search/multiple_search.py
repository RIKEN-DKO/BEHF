
from abc import abstractmethod
from typing import Dict,List
from dkouqe.document_retrieval.semantic_search import SearchContext, print_node_ontology
import numpy as np
import pprint
from dkouqe.document_retrieval.bm25 import BM25
import pickle

class MergeStrategy:
    """Manage how we will combine the search results of several searchs

    """
    @abstractmethod
    def merge_records(self,results):
        return results


class SortByBestBM25Score(MergeStrategy):
    """First score each results by bm25 and then return them sorted.

    """

    def __init__(self,BM25_file,id2index_file ,ascending=True):
        self._ascending = ascending

        print(BM25_file, id2index_file)
        handle = open(BM25_file, 'rb')
        self._bm25 = pickle.load(handle)

        handle = open(id2index_file, 'rb')
        self._id2index = pickle.load(handle)


    @abstractmethod
    def merge_records(self,results,query):
        ##bm25.get_score('melanoma'.split(),id2index['MONDO_0003761'] )
        scores = results['scores']
        ids = results['records_ids']
        new_scores = []
        for id in ids:
            score = self._bm25.get_score(query.split(), self._id2index[id.replace(':','_')])
            new_scores.append(score)

        results['scores'] = new_scores

        return sort_results_by_score(results, self._ascending)

def sort_results_by_score(results,ascending=False):
    scores = results['scores']
    if ascending:
        n = len(scores)
        idx = np.argsort(scores)[::-1][:n]
    else:
        idx = np.argsort(scores)
    #sort all data from the idx
    for key, val in results.items():
        results[key] = np.array(val)[idx]

    return results

class SortByBestScore(MergeStrategy):
    """Manage how we will combine the search results of several searchs

    """
    def __init__(self,ascending=False):
        self._ascending = ascending

    @abstractmethod
    def merge_records(self, results,query):
        return sort_results_by_score(results, self._ascending)



class MultipleRetrievalManager:
    """Handles when we want to retrieve information from multiple index 
    """
    #TODO add type seekers
    def __init__(self, seekers: Dict, merge_strategy: MergeStrategy) -> None:
        """

        :param seekers: A list of objects `SearchContext` to be searcher in. 
        :type seekers: List
        :param merge_strategy: The strategy to be used when merging the results 
            from each SearchContext
        :type merge_strategy: MergeStrategy
        """
        self._merge_strategy = merge_strategy

        #Create a dictionary 
        self._seekers = seekers 
        self._results = self.empty_results() 

    def empty_results(self):
        return {'scores': np.array([]),
                'records_ids': np.array([]),
                # Each record have reference to the seeker object
                # so we can retrieve the name and other info

                'seeker_ref':  np.array([])}

    @property
    def merge_strategy(self)-> MergeStrategy:
        return self._merge_strategy

    @merge_strategy.setter
    def merge_strategy(self, merge_strategy: MergeStrategy) -> None:
        self._merge_strategy = merge_strategy
    

    def retrieve_records(self, query: str, num_docs: int) -> Dict:
        # results = self._results
        #TODO numdocs should be specific for each seeker?
        self._results = self.empty_results()
        for name,seeker in self._seekers.items():
            new_results = seeker.retrieve_records(query,num_docs)
            if 'faiss_scores' in new_results:
                
                self._results['scores'] = np.concatenate(
                    [self._results['scores'], new_results['faiss_scores']])
                self._results['records_ids'] = np.concatenate(
                    [self._results['records_ids'].astype(int),  
                    new_results['records_ids'].astype(int)])
                
                new_refs = np.array([seeker] * len(new_results['faiss_scores']))
                self._results['seeker_ref'] = np.concatenate(
                            [self._results['seeker_ref'], new_refs])

        self._results = self.merge_strategy.merge_records(self._results,query)

        return self._results
    
    def print_results(self) -> None:
        """Search and print the ontology, for debuging.

        Args:
            query (str): [description]
            num_docs (int): [description]
        """

        for i,_ in enumerate(self._results['records_ids']):
            ontology = self._results['seeker_ref'][i]._ontology_graph
            index2id = self._results['seeker_ref'][i]._index2ontoid
            id = self._results['records_ids'][i]
            score = self._results['scores'][i]
            print_node_ontology(id, score, ontology, index2id)


class MultipleOntologiesManager(MultipleRetrievalManager):
    """Handles when we want to retrieve information from multiple index
    """

    def __init__(self,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._results = self.empty_results()

    def empty_results(self):
        return {'scores': np.array([]),
                'records_ids': np.array([]),
                # Each record have reference to the seeker object
                # so we can retrieve the name and other info
                'seeker_ref': [],
                #The ontology nodes for each record
                'graphs': []}

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

                    ontology_graph = seeker._ontology_graph
                    index2id = seeker._index2ontoid
                    nodes = []
                    for id in new_results['records_ids']:
                        node = ontology_graph.nodes[id]
                        nodes.append(node)
                    self._results['nodes'] = np.concatenate(
                                            [self._results['nodes'],nodes])

        self._results = self.merge_strategy.merge_records(self._results, query)

        return self._results

    def print_results(self) -> None:
        """Search and print the ontology, for debuging.

        Args:
            query (str): [description]
            num_docs (int): [description]
        """

        for i, _ in enumerate(self._results['records_ids']):
            id = self._results['records_ids'][i]
            score = self._results['scores'][i]
            node = self._results['nodes'][i]
            print('id ',id)
            print('score ',score)
            pprint.pprint(node)





