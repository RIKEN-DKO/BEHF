# Designing a Semantic-Enhanced Hybrid Search Engine for Biomedical Events

## Introduction

This repository is dedicated to the research and development of the Biomedical Event Hybrid Finder (BEHF), a state-of-the-art semantic search engine. Developed by Julio C Rangel and Norio Kobayashi, BEHF is designed to identify biomedical events within PubMed documents, leveraging advanced technologies like Elasticsearch, SPLADE, and Faiss Approximate Nearest Neighbors (ANN).

## Abstract

The Biomedical Event Hybrid Finder (BEHF) is a semantic search engine tailored for the discovery of biomedical events (BE) in PubMed documents. It combines Elasticsearch's exact match search capabilities with expanded keywords provided by SPLADE, and leverages Faiss ANN for semantic similarity search. BEHF utilizes a unique dataset of network graphs, derived from the PubMed Baseline Database (BD) repository, and offers a flexible scoring mechanism to balance between exact match and semantic similarity searches.




# Creating the pfiles with pubmed titles and abstracts

```
python create_pubmed_dataset.py --dict_file /home/julio/repos/pubmed_processed_data/dicts/pmid2info_full.pickle 
--outdir /home/julio/repos/event_finder/data/pubmed/
```

# Extracting events with deep event mine

Move create directies to Deepeventmine data. And start extracting events on each gpu:

```
sh pubmed.sh e2e rawtext pubmed_1 cg 0
sh pubmed.sh e2e rawtext pubmed_2 cg 1
sh pubmed.sh e2e rawtext pubmed_3 cg 2
sh pubmed.sh e2e rawtext pubmed_4 cg 3

```

# Creating embeddings from events

```
python create_embeddings_sbert.py --indir=/home/julio/repos/ddegk/results/pubmed_small_4/ \
    --outdir=/home/julio/repos/event_finder/data/pubmed_small_graph/  \
    --outname=sbert_embeddings_all-MiniLM-L12-v2.json  \
    --sbertmodel=all-MiniLM-L12-v2 

```

# Creating Faiss index

```
python create_faiss_index.py --indir=../data/pubmed_small_graph/ \
                            --embs_file=sbert_embeddings_all-MiniLM-L12-v2.json \
                            --indexname=faiss_sbert_all-MiniLM-L12-v2

```


## New

Create  graph
```
python -m bef.data.text2events2graphs --yaml data/pubmed_2000s/predict-pubmed_epi.yaml --gpu 0 --start_year 2000 --end_year 2020

python -m bef.data.text2events2graphs --yaml data/pubmed_2000s/predict-pubmed_cg.yaml --gpu 0 --start_year 2020 --end_year 2021
```

Deduplicate the graphs 
```
python -m bef.data.graph_deduplicator data/pubmed_70s/cg data/pubmed_70s/cg
```

Create faiss index:

```
python -m bef.data.graphs2faiss  --graphfile events_graph.json --base_dir data/pubmed_70s/cg


```

Creating the ElasticSearch index:

```
## Don't add '/' at the end 
python -m bef.index.create_elasticsearch_index /home/julio/repos/event_finder/data/pubmed_70s/id/events_graph.json pubmed_70s_id
```

Creating all the faiss and elastic search indices  from dir:
```
 ./create_indices.sh data/pubmed_70s
```


# Batch
Extracting all graphs years from pubmed. 


```
nohup ./text2graphs_new.sh &

```
# Pre-installation
```
$pip install -e .
$git clone git@github.com:jcrangel/dkouqe.git
$cd dkouqe 
$ pip install -e .


pip install git+https://github.com/naver/splade.git
```
# Download data

TODO

## Installing pubmed database: pmid2text
```
python -m bef.create_pmid2text data/pubmed/json_pmid2sents_year
./createdb_pubmed.sh
```


## Running DjangoFront end 
Run hybrid API:

```
python web_app/server_hybrid.py --data_path data/pubmed_70s
```
or
`nohup python web_app/server_hybrid.py --data_path data/pubmed_70s &`

Then run the Django server
```
cd django_frontend
python manage.py runserver 0.0.0.0:8000

```



# Moving to other server

```
git clone git@github.com:jcrangel/event_finder.git
cd event_finder
git clone git@github.com:jcrangel/DeepEventMine_fork.git
mv DeepEventMine_fork/ DeepEventMine
```
