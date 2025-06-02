FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
WORKDIR /app
# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy application code
COPY server.py .
# Expose the port the app runs on
EXPOSE 8881
# Command to run the application
CMD ["python", "server.py"]
