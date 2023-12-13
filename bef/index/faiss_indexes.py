# Defines several funcitons to create several Faiss index types
import numpy as np
import faiss

def create_index_compressed(emb_file, pmids_file, emb_size):
    print("Loading embeddings..")

    embs = np.load(emb_file)
    pmids = np.load(pmids_file)

    # handle = open(emb_file, 'rb')
    # pmid2emb = pickle.load(handle)
    nlist = 100
    k = 4
    ngpus = faiss.get_num_gpus()
    print("Number of GPUs:", ngpus)
    print("Creating index..")
#     quantizer = faiss.IndexFlatL2(emb_size)  # the other index
    #cpu_index = faiss.IndexIVFFlat(quantizer, emb_size, nlist)
    #Compress the vectors
#     cpu_index = faiss.IndexIVFPQ(quantizer, emb_size, nlist,m, 8)

    #https://gist.github.com/mdouze/46d6bbbaabca0b9778fca37ed2bcccf6#file-train_ivf_with_gpu-ipynb
    index2 = faiss.index_factory(emb_size, "PCA64,IVF16384_HNSW32,Flat")
    index_ivf = faiss.extract_index_ivf(index2)
    clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(64))
    index_ivf.clustering_index = clustering_index


#     gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    print("Training..")
#     assert not gpu_index.is_trained
    #Training with gpus
    index2.train(embs)
#     assert gpu_index.is_trained
    print("Adding embdeddings to index..")
    # index2.add_with_ids(np.array(embs), np.array(pmids).astype(np.int))
    index2.add(embs)
    return index2


def create_index_flat(emb_file, ids_file, use_gpu=False,return_ids=False):
    print("Loading embeddings..")

    embeddings = np.load(emb_file)
    ids = np.load(ids_file)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    if use_gpu:
        gpu_index = faiss.index_cpu_to_all_gpus(index)
        gpu_index.add(embeddings)
        return gpu_index

    index.add(embeddings)

    if return_ids:
        return index,ids

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

