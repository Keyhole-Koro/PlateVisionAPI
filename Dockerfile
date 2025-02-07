# Use an official Python image
FROM python:3.8

# Set environment variables for non-interactive installations
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC 

# Install system dependencies for Tesseract and YOLO
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set up a working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy your application code into the container
COPY . /app

# Set the Tesseract language model path inside the container
ENV TESSDATA_PREFIX=/app/model/tesseract/

# Ensure the directory exists
RUN mkdir -p $TESSDATA_PREFIX

# Copy the traineddata files into the correct location
COPY model/tesseract/*.traineddata $TESSDATA_PREFIX
