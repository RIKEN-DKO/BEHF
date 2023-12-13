#!/bin/bash

# Define your database and collection
DATABASE="pmid2text"
COLLECTION="pubMED"

# The input file
JSON_FILE="/home/julio/repos/event_finder/data/pubmed_70s/pmid2text.json"

# Temp file for processing
TEMP_FILE="temp.json"

# Replace newline characters with space and create objects for each key-value pair
cat "$JSON_FILE" | tr '\n' ' ' | jq -c 'to_entries[] | {"_id": .key, "value": .value}' > "$TEMP_FILE" 2> /dev/null

# Import to MongoDB
mongoimport --db $DATABASE --collection $COLLECTION --type json --file "$TEMP_FILE" 

# Remove temp file
rm "$TEMP_FILE"
