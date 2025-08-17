# Use a lightweight Python base image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required by scikit-learn and other packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download required NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy the rest of the application code
COPY . .

# Create necessary directories
RUN mkdir -p models results logs

# Expose Flask port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app/app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app/src

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Run the Flask app
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
