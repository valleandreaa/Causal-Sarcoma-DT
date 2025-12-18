FROM python:3.12-slim

# ENV DEBIAN_FRONTEND=noninteractive

ARG DEVICE_TYPE=cpu


COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt


WORKDIR /app

# RUN mkdir -p /app/Models

# # Copy the entire current working directory to /app/Models
# COPY train/ /app/Models/train/
# COPY test/ /app/Models/test/
# COPY data/ /app/Models/data/
# COPY configs/ /app/Models/configs/
# COPY evaluate/ /app/Models/evaluate/
# COPY metrics/ /app/Models/metrics/
# COPY models/ /app/Models/models/
# COPY monitoring/ app/Models/monitoring/
# COPY scripts/ /app/Models/scripts/
# COPY utils/ /app/Models/utils/
# COPY .env/  /app/Models/.env

ENV PYTHONUNBUFFERED=1
