# server/services/llm_service.py
import os
from langchain_aws.chat_models.bedrock import ChatBedrock

def invoke_llm(input_text):
    """
    Use LLM (Amazon Bedrock) to analyze user input and determine mood/sentiment.
    """
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    llm = ChatBedrock(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        model_id="amazon.titan-text-lite-v1",
        region_name="us-west-2"
    )
    
    response = llm.invoke(input_text)
    return response.content  # This will return the sentiment or mood
