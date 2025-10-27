#!/bin/bash

# Wait for LocalStack to be ready
echo "Waiting for LocalStack to be ready..."
until aws --endpoint-url=http://localstack:4566 s3 ls > /dev/null 2>&1
do
    echo "Waiting for LocalStack..."
    sleep 2
done

# Create the buckets
echo "Creating S3 buckets..."
awslocal s3 mb s3://gst-training || echo "Bucket gst-training already exists"
awslocal s3 mb s3://bookiewand-ai || echo "Bucket bookiewand-ai already exists"

echo "S3 setup complete!"