import boto3
from botocore.exceptions import ClientError
import json


import os
from dotenv import load_dotenv  #  pip3 install python-dotenv
from configparser import ConfigParser, ExtendedInterpolation


# Read config.ini file
# tests_dir = os.path.dirname(__file__)
# root_dir = os.path.abspath(os.path.join(tests_dir, os.pardir))
# config_dir = os.path.join(root_dir, 'config')
config_dir = "config"
load_dotenv(dotenv_path=os.path.join(config_dir, ".env"))
config_file = os.environ["CONFIG_FILE"]
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(f"{config_dir}/{config_file}")


# Initiate session
session = boto3.Session(
    aws_access_key_id=config["BedRock_LLM_API"]["aws_access_key_id"],
    aws_secret_access_key=config["BedRock_LLM_API"]["aws_secret_access_key"],
    aws_session_token=config["BedRock_LLM_API"]["aws_session_token"],
)
bedrock_client = session.client("bedrock-runtime", region_name="us-east-1")


# Set the model ID, e.g., Titan Text Premier.
model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

content = "What is the capital of france?"

body_content = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1000,
    "temperature": 0,
    "messages": [{"role": "user", "content": content}],
}


# Replace 'YourSonnetAPIEndpoint' with the actual endpoint for the Sonnet API
response = bedrock_client.invoke_model(modelId=model_id, body=json.dumps(body_content))
response_body = response["body"].read().decode()
print(response_body)
