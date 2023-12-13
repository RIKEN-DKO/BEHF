
import numpy as np
import pprint
import time
from bef.search.vector_utils import transform_and_normalize

def vector_search(query, model, index, num_results=10, nprobe=1):
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
    # if redu_dim > 0:


    # index.nprobe = nprobe
    D, I = index.search(np.array(vector).astype("float32"), k=num_results)
    return D, I


def search_and_print(user_query, model, index, graph):

    start = time.time()
    distances, ids = vector_search(
        [user_query], model, index, num_results=10, nprobe=10)
    distances = distances.flatten().tolist()
    print(distances)
    print(ids)
    ids = ids.flatten().tolist()
    print('Searching took: ', time.time()-start)
    # print(
    #     f'L2 distance: {D.flatten().tolist()}\n\n IDs: {I.flatten().tolist()}')

    # for i in range(len(ids)-1, 0, -1):
    for i in range(len(ids)):
        id_ = ids[i]
        dist = distances[i]
        print('id: ', id_, 'distance: ', dist)
        pprint.pprint(graph[id_]['nodes'])
        print('')
        # print json.dumps(graph.nodes[index2ontoid[i]], indent=4)


def search_apply_kernel(user_query, model, index, graph,kernel,bias):


    start = time.time()


    vector = model.encode([user_query])

    vector = transform_and_normalize(vector, kernel, bias)
    distances, ids = index.search(vector.astype("float32"), k=10)

    distances = distances.flatten().tolist()
    print(distances)
    print(ids)
    ids = ids.flatten().tolist()
    print('Searching took: ', time.time()-start)
    # print(
    #     f'L2 distance: {D.flatten().tolist()}\n\n IDs: {I.flatten().tolist()}')

    # for i in range(len(ids)-1, 0, -1):
    for i in range(len(ids)):
        id_ = ids[i]
        dist = distances[i]
        print('id: ', id_, 'distance: ', dist)
        pprint.pprint(graph[id_]['nodes'])
        print('')

