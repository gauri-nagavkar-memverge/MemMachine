#!/bin/bash

apt update
apt install -y docker.io docker-compose python3-pip
pip install awscli --break-system-packages
cd /tmp

OPENAI_API_KEY=$(aws secretsmanager get-secret-value   --secret-id memmachine/openai-key --region us-east-1 --query SecretString   --output text)
echo "OPENAI_API_KEY=$OPENAI_API_KEY" >> .env
echo "WEBUI_URL=https://playground.memmachine.ai" >> .env
echo "GOOGLE_REDIRECT_URI=https://playground.memmachine.ai/oauth/google/callback" >> .env
GOOGLE_CLIENT_ID=$(aws secretsmanager get-secret-value   --secret-id memmachine/google/client_id --region us-east-1 --query SecretString   --output text)
echo "GOOGLE_CLIENT_ID=$GOOGLE_CLIENT_ID" >> .env
GOOGLE_CLIENT_SECRET=$(aws secretsmanager get-secret-value   --secret-id memmachine/google/client_secret --region us-east-1 --query SecretString   --output text)
echo "GOOGLE_CLIENT_SECRET=$GOOGLE_CLIENT_SECRET" >> .env


aws s3 cp s3://my-memmachine-configs/docker-compose.yml .
aws s3 cp s3://my-memmachine-configs/configuration.yml .

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 114147526645.dkr.ecr.us-east-1.amazonaws.com
docker-compose pull
docker-compose up -d