FROM nvcr.io/nvidia/pytorch:23.10-py3

WORKDIR /app

# Copy dependency requirements
COPY requirements.txt .

# Install pinned Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy server implementation code
COPY server.py .

# Expose Uvicorn API port
EXPOSE 8000

# Set production execution entrypoint
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]