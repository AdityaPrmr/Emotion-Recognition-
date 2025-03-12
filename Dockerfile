# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -i https://pypi.org/simple -r requirements.txt

# Copy the rest of the app's code
COPY . .

# Expose port 5000 for Flask
EXPOSE 10000

# Command to run the application
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:$PORT", "app:app"]
