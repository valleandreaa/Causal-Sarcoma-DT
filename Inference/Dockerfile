FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all inference scripts
COPY . .

# Expose ports for FastAPI, Streamlit, and Jupyter
EXPOSE 8000 8501 8888

# Default command to run FastAPI, Streamlit, and Jupyter in parallel
# CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port 8000 & streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0 & jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.token='securetoken'"]
