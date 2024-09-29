from langchain_aws.chat_models.bedrock import ChatBedrock
import os

AWS_ACCESS_KEY_ID = os.getenv("aws_access_key_id")
AWS_SECRET_ACCESS_KEY = os.getenv("aws_secret_access_key")

def invokeLLM(input):
    # Initialize the Bedrock LLM
    llm = ChatBedrock(aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY, model_id="amazon.titan-text-lite-v1", region_name="us-west-2")

    # Use the model for text generation
    prompt = input
    response = llm.invoke(prompt)
    return response.content

invokeLLM("hello")
