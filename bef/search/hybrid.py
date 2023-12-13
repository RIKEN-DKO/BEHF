

class HybridSeeker:
    """
    Handles hybrid searches using both dense and sparse handlers.

    The class enables hybrid searches using both dense (FAISS) and sparse (Elasticsearch) search methods. 
    It also provides functionality for retrieving document annotations.

    Attributes:
        dense_handler: Object that performs dense (FAISS) search.
        sparse_handler: Object that performs sparse (Elasticsearch) search.
        normalization (bool): Determines whether to normalize the scores during hybrid search.

    Args:
        dense_handler: The handler for dense search.
        sparse_handler: The handler for sparse search.
        normalization (bool, optional): If True, normalization is applied during the hybrid search.
            Default is True.
    """
    def __init__(self, dense_handler, sparse_handler,normalization=True):
        self.dense_handler = dense_handler
        self.sparse_handler = sparse_handler
        self.normalization = normalization

    def search(self, query, num_res, type="search", alpha=0.1, weight_on_dense=False):
        """
        Performs a hybrid search using both dense and sparse search methods.

        Args:
            query (str): The query string.
            num_res (int): The number of search results to retrieve.
            type (str, optional): The type of search to perform. Default is "search".
            alpha (float, optional): The weight for the sparse handler in the hybrid score computation. 
                Default is 0.1.
            weight_on_dense (bool, optional): If True, weights are calculated based on dense results. 
                Default is False.

        Returns:
            dict: A dictionary containing the query and the hybrid search results.
        """
        if alpha == 0:
            dense_hits = self.dense_handler(query, num_res, type)['results']
            sparse_hits = []
            print('Only dense')
        elif alpha == 1:    
            print('Only sparse')
            sparse_hits = self.sparse_handler(query,  type)['results']
            dense_hits = []
        else:
            dense_hits = self.dense_handler(query, num_res, type)['results']
            sparse_hits = self.sparse_handler(query,  type)['results']
        
        for hit in sparse_hits:
            hit['search_type']='sparse'
        for hit in dense_hits:
            hit['search_type']='dense'

        return {'query':query,
                'results':self._hybrid_results(dense_hits, sparse_hits, alpha, num_res, weight_on_dense)}

    def _hybrid_results(self,dense_results, sparse_results, alpha, num_res,weight_on_dense=False):
        """
        Computes hybrid results from dense and sparse search results.

        Args:
            dense_results (list): List of results from dense search.
            sparse_results (list): List of results from sparse search.
            alpha (float): The weight for the sparse handler in the hybrid score computation.
            num_res (int): The number of search results to retrieve.
            weight_on_dense (bool): If True, weights are calculated based on dense results.

        Returns:
            list: A list of dictionaries, each containing information about a search hit and its hybrid score.
        """
        dense_hits = {hit['graph_id']: hit for hit in dense_results}
        sparse_hits = {hit['graph_id']: hit for hit in sparse_results}
        print('dense hits:',len(dense_hits.values()))
        print('sparse hits:',len(sparse_hits.values()))

        
        # If either the dense_hits or sparse_hits are empty, return the non-empty hits
        if len(dense_hits) == 0:
            # print('only sparse')
            return list(sparse_hits.values())
        elif len(sparse_hits) == 0:
            # print('only dense')
            return list(dense_hits.values())


        hybrid_result = []
        epsilon = 0.00001 # to prevent division by zero

        min_dense_score = min(hit['total_score'] for hit in dense_hits.values()) if len(dense_hits) > 0 else 0
        max_dense_score = max(hit['total_score'] for hit in dense_hits.values()) if len(dense_hits) > 0 else 1
        min_sparse_score = min(hit['total_score'] for hit in sparse_hits.values()) if len(sparse_hits) > 0 else 0
        max_sparse_score = max(hit['total_score'] for hit in sparse_hits.values()) if len(sparse_hits) > 0 else 1

        # print(min_dense_score,max_dense_score,min_sparse_score,max_sparse_score)
        for graph_id in set(dense_hits.keys()) | set(sparse_hits.keys()):
            if graph_id not in dense_hits:
                sparse_score = sparse_hits[graph_id]['total_score']
                dense_score = min_dense_score
            elif graph_id not in sparse_hits:
                sparse_score = min_sparse_score
                dense_score = dense_hits[graph_id]['total_score']
            else:
                sparse_score = sparse_hits[graph_id]['total_score']
                dense_score = dense_hits[graph_id]['total_score']
            # print(graph_id)


            # Apply normalization
            if self.normalization:

                sparse_score = (sparse_score - (min_sparse_score + max_sparse_score) / 2) \
                                / (max_sparse_score - min_sparse_score + epsilon)

                dense_score = (dense_score - (min_dense_score + max_dense_score) / 2) \
                                / (max_dense_score - min_dense_score + epsilon)

            score = alpha * sparse_score + (1-alpha)*dense_score 
            hybrid_result.append({**dense_hits.get(graph_id, {}), **sparse_hits.get(graph_id, {}), 'total_score': score})

        # Sort the results by the new total_score
        sorted_results = sorted(hybrid_result, key=lambda x: x['total_score'], reverse=True)[:num_res]

        # Recompute the rank based on the new order
        for i, result in enumerate(sorted_results, 1):
            result['rank'] = i

        return sorted_results
    

    def get_doc_annotations(self, doc_id):
        """
        Retrieves document annotations for a specific document.

        Args:
            doc_id (str): The ID of the document to retrieve annotations for.

        Returns:
            dict or None: A dictionary containing document annotations if found, else None.
        """
        return self.dense_handler.get_doc_annotations(doc_id)



