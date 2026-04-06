# Use stable Python
FROM python:3.11-slim

# Prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1

# Install system dependencies (ffmpeg + basics)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency file first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Create required directories (important for your app!)
RUN mkdir -p web_uploads generated_videos audio_files

# Expose port (Render uses $PORT but this is good practice)
EXPOSE 10000

# Start your Flask app via gunicorn
CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 360