#!/bin/bash

# AWS SES Local Setup Script
# This script initializes SES using LocalStack for local testing

echo "Setting up SES using LocalStack..."

# Wait for LocalStack to be ready
echo "Waiting for LocalStack to be ready..."
until aws --endpoint-url=http://localhost:4566 ses get-send-quota --region us-east-1 > /dev/null 2>&1; do
    echo "LocalStack not ready yet, waiting..."
    sleep 2
done

echo "SES local is ready!"

# Verify email addresses in SES (aws-ses-v2-local requires verification)
echo "Verifying test email addresses..."

# Verify sender email
aws --endpoint-url=http://localhost:4566 ses verify-email-identity \
    --email-address noreply@bookiewand.com \
    --region us-east-1

# Verify recipient email for testing
aws --endpoint-url=http://localhost:4566 ses verify-email-identity \
    --email-address test@bookiewand.com \
    --region us-east-1

# Verify support email
aws --endpoint-url=http://localhost:4566 ses verify-email-identity \
    --email-address support@bookiewand.com \
    --region us-east-1

echo "Email addresses verified!"

# Get SES send quota
echo "SES Send Quota:"
aws --endpoint-url=http://localhost:4566 ses get-send-quota --region us-east-1

# List verified identities
echo "Verified Email Identities:"
aws --endpoint-url=http://localhost:4566 ses list-verified-email-addresses --region us-east-1

echo "SES setup complete!"
