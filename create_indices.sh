#!/bin/bash

#Check if parameter was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 base_dir"
    exit 1
fi

base_dir="$1"
out_file="creating_index.out"

#Get total number of subdirectories for progress
total_dirs=$(find "$base_dir" -type d -mindepth 1 -maxdepth 1 | wc -l)
count=0

# Iterate over each subdirectory of the base directory
for sub_dir in "$base_dir"/*; do
    if [ -d "$sub_dir" ]; then
        count=$((count+1))
        
        # Convert sub_dir path to Elasticsearch index by replacing '/' with '_'
        es_index=$(echo "$sub_dir" | tr '/' '_')

        #Creating graphs database
        echo "#Creating graphs database for $sub_dir"
        echo "python -m bef.data.graph_deduplicator $sub_dir $sub_dir" | tee -a $out_file
        python -m bef.data.graph_deduplicator $sub_dir $sub_dir 2>&1 | tee -a $out_file

        #Creating Faiss index
        echo "#Creating Faiss index"
        echo "python -m bef.data.graphs2faiss --graphfile events_graph.json --base_dir $sub_dir" | tee -a $out_file
        python -m bef.data.graphs2faiss --graphfile events_graph.json --base_dir $sub_dir 2>&1 | tee -a $out_file

        #Creating Elastic search index
        echo "#Creating Elastic search index"
        echo "python -m bef.index.create_elasticsearch_index $sub_dir/events_graph.json $es_index" | tee -a $out_file
        python -m bef.index.create_elasticsearch_index $sub_dir/events_graph.json $es_index 2>&1 | tee -a $out_file

        #Print progress
        echo "Processed $count out of $total_dirs directories."
    fi
done

echo "Done. Check the $out_file for the output of each command."
