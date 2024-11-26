# Use an official Python slim image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the project files to the container
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --index-url https://pypi.org/simple

# Expose port (optional if a server runs)
# EXPOSE 5000

# Command to run the Python script
CMD ["python", "gesture_recognition.py"]
