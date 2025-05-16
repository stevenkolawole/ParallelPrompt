#!/bin/bash

# Set AWS credentials and region
export AWS_KEY="YOUR_AWS_KEY"
export AWS_SECRET_KEY="YOUR_AWS_SECRET_KEY"
export AWS_REGION="YOUR_AWS_REGION"
echo "AWS environment variables set."

# Run the script
python3 find_parallelprompts.py --dataset lmsys/lmsys-chat-1m
echo "Script executed."