# Defines several funcitons to create several Faiss index types
import numpy as np
import faiss

def create_index_compressed(embs, ids=None):
    """
    Create a compressed Faiss index using the provided embeddings and IDs.

    :param embs: Embeddings to be indexed.
    :type embs: numpy array
    :param ids: IDs corresponding to the embeddings.
    :type ids: numpy array or None

    :return: Compressed Faiss index.
    """
    emb_size = embs.shape[1]
    print("Loading embeddings..")

    nlist = min(16384, len(embs))  # Ensure nlist is <= number of embeddings
    k = 4
    ngpus = faiss.get_num_gpus()
    print("Number of GPUs:", ngpus)
    print("Creating index..")
    res = faiss.StandardGpuResources()  # Use standard GPU resources
    index2 = faiss.index_factory(emb_size, "PCA64,IVF{}_HNSW32,Flat".format(nlist))
    index_ivf = faiss.extract_index_ivf(index2)
    clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(64))
    index_ivf.clustering_index = clustering_index

    # Convert the index to GPU
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index2)

    print("Training..")
    index_gpu.train(embs)
    print("Adding embeddings to index..")
    if ids is not None:
        index_gpu.add_with_ids(embs, ids)
    else:
        index_gpu.add(embs)
    return index_gpu



def create_index_flat(embeddings, ids=None, use_gpu=False):
    print("Loading embeddings..")

        #not tested
    if use_gpu:
        gpu_index = faiss.index_cpu_to_all_gpus(index)
        gpu_index.add(embeddings)
        return gpu_index

    if ids is not None:
        index_flat = faiss.IndexFlatL2(embeddings.shape[1])
        ids = np.array(ids,dtype='int64')
        index = faiss.IndexIDMap(index_flat)
        index.add_with_ids(embeddings, ids)

    else:
        index= faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)


    return index


def create_index_IVF(emb_file, use_quantization=False, metric=faiss.METRIC_L2):
    """
    https://towardsdatascience.com/understanding-faiss-619bb6db2d1a
    We use the ‘IndexIVFFlat’ index type for our vectors. 
    The ‘Flat’ here signifies that the vectors are stored as 
    s without any compression or quantisation (more on that later). The IVF index takes two parameters:

    nlist : to specify the number of clusters to be formed
    quantizer : to assign the vectors to a particular cluster. 
    This is usually another index that uses the L2 distance metric (we use the FlatL2 index)
    """
    print("Loading embeddings..")

    embeddings = np.load(emb_file)
    d = embeddings.shape[1]
    nlist = 5  # number of clusters
    #                            dimension
    quantiser = faiss.IndexFlatL2(d)

    if use_quantization:
        #The vectors are still stored in Voronoi cells,
        # but their size is reduced to a configurable number of bytes m (d must be a multiple of m).
        m = 128  # bigger m gives bigger index file
        #ndex *quantizer, size_t d, size_t nlist, size_t M,
        # size_t nbits_per_idx, MetricType metric = METRIC_L2)
        # 8 specifies that each sub-vector is encoded as 8 bits
        index = faiss.IndexIVFPQ(
            quantiser, d, nlist, m, 8,metric)
    else:
        index = faiss.IndexIVFFlat(
            quantiser, d, nlist,   metric)

    print(index.is_trained)   # False
    index.train(embeddings)  # train on the database vectors
    print(index.ntotal)   # 0
    index.add(embeddings)   # add the vectors and update the index
    print(index.is_trained)  # True
    print(index.ntotal)   # un buen

    return index


#https://github.com/facebookresearch/faiss/blob/master/benchs/bench_hnsw.py


def create_index_HNSW(emb_file, metric=faiss.METRIC_L2):

    print("HNSW ")
    embeddings = np.load(emb_file)
    d = embeddings.shape[1]
    M = 32
    #M is the number of neighbors used in the graph. A larger M is more accurate but uses more memory
    index = faiss.IndexHNSWFlat(d, M)

    # training is not needed

    # this is the default, higher is more accurate and slower to
    # construct
    index.hnsw.efConstruction = 40

    print("add")
    # to see progress
    index.verbose = True
    index.add(embeddings)
    return index

