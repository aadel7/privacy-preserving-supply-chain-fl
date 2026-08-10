# Use the official Python 3.10 slim image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the server application code
COPY . .

# Expose the port the Flower server will run on
EXPOSE 8080

# Command to run the Flower aggregation server
CMD ["python", "main.py"]
