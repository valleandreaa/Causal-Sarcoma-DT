FROM python:3.12


# set the working directory
WORKDIR /app

# check if i need to remove something here
COPY dags/ /app/dags/
COPY requirements.txt /app/requirements.txt
COPY etl_sdt/ /app/etl_sdt
COPY data/ /app/data
COPY pipelines/ /app/pipelines

# install any required packages
RUN pip install -r /app/requirements.txt

# set PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app/etl_sdt"
