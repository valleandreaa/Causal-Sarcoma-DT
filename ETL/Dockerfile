FROM apache/airflow:2.9.3-python3.12


# Set the working directory
WORKDIR /usr/local/airflow

# Copy DAGs and local packages
COPY dags/ /usr/local/airflow/dags/
COPY requirements.txt/ /usr/local/airflow/requirements.txt
COPY etl_sdt/ /usr/local/airflow/etl_sdt
COPY data/ /usr/local/airflow/data
COPY pipelines/ /usr/local/airflow/pipelines

# Install any required packages
RUN pip install apache-airflow==${AIRFLOW_VERSION} -r /usr/local/airflow/requirements.txt

# Set PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/usr/local/airflow/etl_sdt"
