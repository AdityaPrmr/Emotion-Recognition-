# Use PyTorch's official CPU image (ensures compatibility)
FROM pytorch/pytorch:latest

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV, Librosa, and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install required Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose port required for Flask
EXPOSE 10000

# Start the application with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
