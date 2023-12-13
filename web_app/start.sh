#!/bin/bash

function cleanup() {
    kill $(jobs -p)
    kill -s SIGINT $app_pid
}

trap 'cleanup' SIGINT SIGTERM EXIT

python server.py --port 5556 --event id &
python server.py --port 5555 --event cg &
streamlit run app.py --server.fileWatcherType none &

app_pid=$!
echo "All services up"

wait
