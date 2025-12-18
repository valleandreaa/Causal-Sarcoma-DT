#!/bin/bash

# Step 3: Get the scheduler container ID
SCHEDULER_CONTAINER_ID=$(docker ps -qf "name=airflow-scheduler")

# Step 4: Access the scheduler container and create a writable cache directory
docker exec -it $SCHEDULER_CONTAINER_ID /bin/bash -c "mkdir -p /home/airflow/writable_cache"

# Step 5: Set the TRANSFORMERS_CACHE environment variable and run the DAG task in test mode
docker exec -it $SCHEDULER_CONTAINER_ID /bin/bash -c "export TRANSFORMERS_CACHE=/home/airflow/writable_cache"

# Step 6: Check the output in the terminal
docker logs $SCHEDULER_CONTAINER_ID