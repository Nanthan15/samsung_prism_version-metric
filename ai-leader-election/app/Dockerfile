# Dockerfile
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 80

# Run the app
CMD ["python", "app.py","collect_metrix.py"]

# Copy the script into the container
COPY collect_metrix.py .

