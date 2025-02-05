# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port your app will run on (e.g., 8080 for Flask/FastAPI)
EXPOSE 8081

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8081"]
