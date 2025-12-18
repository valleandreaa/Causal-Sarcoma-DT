# Airflow ETL Sarcoma Data Pipeline

This repository contains the setup for an ETL process for Sarcoma data using Apache Airflow.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup Instructions](#setup-instructions)
   - [Build the Docker Image](#build-the-docker-image)
   - [Run the Containers](#run-the-containers)
   - [Access Airflow Web Interface](#access-airflow-web-interface)
3. [Testing](#testing)
   - [List Available DAGs](#list-available-dags)
   - [Run DAG in Test Mode](#run-dag-in-test-mode)

## Prerequisites

- Docker
- Docker Compose

## Setup Instructions

### Build the Docker Image

1. Clone the repository:

```sh
git clone https://github.com/valleandreaa/Sarcoma-DT.git
cd ETL
```

2. Environment Variables
Ensure the following environment variables are stored in the .env file
```sh
AIRFLOW_UID=1000
AIRFLOW_GID=0
MONGO_URI=mongodb://root:example@localhost:28017
MONGO_DB=SarcomaDB
MONGO_COLLECTION=PatientsTest
```
2. Ensure the Dockerfile and docker-compose.yml are in the root directory.

3. Build the Docker image:

```sh
docker build -t etl_sarcoma_dt .
```

### Run the Containers

To run just the mongodb a special docker compose is required:

```sh
docker compose -f docker-compose.mongo.yaml up mongodb
```
This is useful for testing. And remember to generate the DB and the collection (use MongoDB Compass)

If you want to run the compleet service go with:
1. Start the Docker Compose services:

```sh
docker compose up -d
```

2. Verify the services are running:

```sh
docker compose ps
```

### Set cache enviroment

1. Make the script executable:

```sh
chmod +x system/env_cache_set.sh
```

2. Run the script:

   ```sh
   system/env_cache_set.sh
   ```


### Access Airflow Web Interface

1. Open your web browser and go to `http://localhost:8080`.
2. You should see the Airflow web interface.

## Testing

### List Available DAGs

1. Access the scheduler container:

   ```sh
   docker exec -it <scheduler-container-id> /bin/bash
   ```

   Replace `<scheduler-container-id>` with the actual ID or name of your scheduler container. You can find this using:

   ```sh
   docker ps
   ```

2. List available DAGs:

   ```sh
   airflow dags list
   ```

### Run DAG in Test Mode

1. Access the scheduler container:

   ```sh
   docker exec -it <scheduler-container-id> /bin/bash
   ```

2. Run the DAG task in test mode:

   ```sh
   airflow tasks test etl_sarcoma_dt extract_data 2023-07-27
   ```

3. Check the output in the terminal to ensure the task runs as expected.
   ```sh
   docker exec -it <webserver_container_id> airflow dags trigger <your_dag_id>
   ```


### Dump MongoDB Database

1. Install MongoDB Database Tools:

   ```sh
   sudo apt-get install -y mongodb-database-tools
   ```

2. Remove the existing db folder to ensure a new backup:

   ```sh
   rm -rf Models/db
   ```
3. Run mongodump to create a backup of the SarcomaDB database:

   ```sh
   mongodump --db SarcomaDB --out /home/andrea/Desktop/Sarcoma-DT/Models/dump
   ```

## Deployment Lambda Labs Instructions

1. Export the db from the db into the container:
   ```sh
      docker exec -it mongodb mongodump --authenticationDatabase admin --username root --password example --db SarcomaDB --out /sarcoma_backup/
   ```

2. Copy the db backup from my container to the local machine:


   ```sh
      docker cp mongodb:/sarcoma_backup/ ./sarcoma_backup/
   ```
3. Copy from local machine to the Lambda machine:

   ```sh
      scp -r ./sarcoma_backup/ andreavalle@10.180.48.11:/home/andreavalle/Sarcoma-DT/ETL
   ```

4. Copy db backup in the container:

   ```sh
      docker cp ~/S/ETL/sarcoma_backup/ mongodb:/sarcoma_backup
   ```

5. Restore the collections manually, e.g.:
   ```sh
      docker exec -it mongodb mongorestore --authenticationDatabase admin --username root --password example --db SarcomaDB --collection PatientsTest_2 /sarcoma_backup/SarcomaDB/PatientsTest_2.bson
   ```
