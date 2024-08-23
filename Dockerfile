# Base img
FROM python

# Working Directory
WORKDIR /mlapp

# Copy Requirments.txt and app.py, and data/rental_1000.csv
COPY . .

#Libraries
RUN pip install --no-cache-dir -r requirements.txt

# Default Commands to run at start of Container
CMD ["python","app.py"]

