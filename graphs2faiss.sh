#!/bin/bash
        # last 17 + 3 
CUDA_VISIBLE_DEVICES=1
for i in {2000..2015} ;do
    c=$((i))
    BASE_DIR="data/pubmed_full/results/id/"
    GRAPH_FILE="graph_pubmed_$c.json"
    echo $GRAPH_FILE
    # python -m bef.text2graphs --yaml data/pubmed_full/predict-pubmed.yaml --gpu 0 --json_file $FILE &
    python -m bef.graphs2faiss  --graphfile  $GRAPH_FILE --base_dir $BASE_DIR
done


echo "All done"