#!/bin/bash

# Function to process and import JSON files into MongoDB
import_to_mongo() {
    JSON_FILE=$1
    # Extract filename without extension
    FILENAME=$(basename "$JSON_FILE" .json)
    # Extract the immediate parent directory name
    PARENT_DIR=$(basename $(dirname "$JSON_FILE"))
    # Form the database name by combining the parent directory name and the filename
    DATABASE="${PARENT_DIR}_${FILENAME}"
    COLLECTION="data"

    TEMP_FILE="temp.json"

    # Check if the JSON is an array or an object
    FIRST_CHAR=$(head -c 1 "$JSON_FILE")

    if [ "$FIRST_CHAR" == "[" ]; then
        # If it's an array, process it differently
        cat "$JSON_FILE" | tr '\n' ' ' | jq -c '.[] | {"_id": .id, "value": .}' > "$TEMP_FILE" 2> /dev/null
    else
        # If it's an object, process it as in the original script
        cat "$JSON_FILE" | tr '\n' ' ' | jq -c 'to_entries[] | {"_id": .key, "value": .value}' > "$TEMP_FILE" 2> /dev/null
    fi

    # Import to MongoDB
    mongoimport --db $DATABASE --collection $COLLECTION --type json --file "$TEMP_FILE" 

    # Remove temp file
    rm "$TEMP_FILE"
}

# Check if directory argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <base_directory>"
    exit 1
fi

BASE_DIR=$1

# Call the function for each of your JSON files
import_to_mongo "$BASE_DIR/events_graph.json"
import_to_mongo "$BASE_DIR/graphs2docs.json"
import_to_mongo "$BASE_DIR/docs2graphspans.json"
