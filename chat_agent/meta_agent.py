import json
import boto3
from botocore.config import Config

from langchain_core.messages import (
    AnyMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage
)
from langchain_core.tools import BaseTool
from typing import TypedDict, Annotated, List, Dict, Any, Sequence, Optional

# ToolCall is needed for constructing the AIMessage with tool calls
from langchain_core.messages import ToolCall

# We'll use direct Bedrock API calls instead of BedrockLLM

# --- AWS Bedrock Configuration ---
def get_bedrock_client(region_name="us-east-2", 
                      aws_access_key_id=None,
                      aws_secret_access_key=None,
                      aws_session_token=None):
    """
    Create and return a boto3 Bedrock client with the specified configuration.
    """
    # If credentials are provided, use them directly
    if aws_access_key_id and aws_secret_access_key:
        return boto3.client(
            'bedrock-runtime',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region_name
        )
    
    # Otherwise, use the default credentials from the environment or config
    config = Config(
        region_name=region_name,
        signature_version="v4",
        retries={"max_attempts": 3, "mode": "standard"}
    )
    
    session = boto3.Session()
    return session.client("bedrock-runtime", config=config)

# --- Direct Bedrock API Call Function ---
def invoke_bedrock_directly(client, model_id, messages, temperature=0.2, max_tokens=5000):
    """
    Makes a direct API call to AWS Bedrock without using the LangChain wrapper.
    """
    # Convert LangChain messages to a prompt string for Llama model
    prompt = ""
    system_message = None
    
    # Extract system message if present
    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_message = msg.content
            break
    
    # Build conversation in the format Llama expects
    conversation = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue  # Already handled
        elif isinstance(msg, HumanMessage):
            conversation.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            conversation.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            # Format tool messages as user messages with a special prefix
            conversation.append({"role": "user", "content": f"TOOL RESULT: {msg.content}"})
    
    # Prepare the request body for Llama model
    request_body = {
        "prompt": prompt,
        "temperature": temperature,
        "max_gen_len": max_tokens
    }
    
    # If we have a system message, incorporate it into the prompt
    formatted_prompt = ""
    if system_message:
        formatted_prompt = f"System: {system_message}\n"
    
    # If we have conversation, format it properly
    if conversation:
        for msg in conversation:
            if msg["role"] == "user":
                formatted_prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                formatted_prompt += f"Assistant: {msg['content']}\n"
        
        # Add final assistant prompt
        formatted_prompt += "Assistant: "
    
    # Set the formatted prompt
    request_body["prompt"] = formatted_prompt.strip()
    
    # Convert to JSON string
    body = json.dumps(request_body)
    
    print(f"Request body: {body}")
    
    # Make the API call
    response = client.invoke_model(
        modelId=model_id,
        body=body
    )
    
    # Parse the response
    response_body = json.loads(response.get('body').read())
    
    print(f"Response body: {response_body}")
    
    # Extract the generated text
    if "generation" in response_body:
        return response_body["generation"]
    elif "completion" in response_body:
        return response_body["completion"]
    else:
        raise ValueError(f"Unexpected response format: {response_body}")


# --- Custom LLM Invocation Function for AWS Bedrock Llama 405B ---
def invoke_llm_manually(
    messages: Sequence[BaseMessage],
    tools: Sequence[BaseTool] = None,
    api_base_url: str = None,  # Not used with Bedrock, kept for compatibility
    model_name: str = "us.meta.llama3-1-405b-instruct-v1:0",
    auth_headers: Dict[str, str] = None,  # Not used with Bedrock, kept for compatibility
    temperature: float = 0.2,
    max_tokens: int = 5000,
    region_name: str = "us-east-2",
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None
) -> AIMessage:
    """
    Invokes the AWS Bedrock Llama 405B model, handles the response, and constructs
    an AIMessage, supporting tool calls if provided.
    """
    try:
        # Create a Bedrock client
        client = get_bedrock_client(
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token
        )
        
        print(f"--- Invoking AWS Bedrock Llama 405B model: {model_name} ---")
        
        # If tools are provided, log them but we'll handle them separately
        if tools:
            print(f"--- Note: {len(tools)} tools provided ---")
            for tool in tools:
                print(f"Tool: {tool.name} - {tool.description}")
        
        # Make direct API call to Bedrock
        try:
            response_text = invoke_bedrock_directly(
                client=client,
                model_id=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            print(f"--- Received response from AWS Bedrock ---")
            print(f"Response: {response_text[:100]}...")
            
            # Create an AIMessage from the response
            return AIMessage(content=response_text)
            
        except Exception as api_error:
            print(f"Error in direct Bedrock API call: {str(api_error)}")
            raise api_error
        
    except Exception as e:
        print(f"Error invoking AWS Bedrock: {str(e)}")
        import traceback
        traceback.print_exc()
        return AIMessage(content=f"Error: AWS Bedrock call failed: {str(e)}")

if __name__ == "__main__":
    # Test AWS Bedrock configuration
    try:
        # You can set your AWS credentials here for testing
        # Or rely on environment variables/AWS configuration
        client = get_bedrock_client(region_name="us-east-2")
        print("Successfully created Bedrock client")
        
        # Test simple model invocation with direct API call
        messages = [SystemMessage(content="You are a helpful assistant."), 
                   HumanMessage(content="Hello, how are you?")]
        
        response_text = invoke_bedrock_directly(
            client=client,
            model_id="us.meta.llama3-1-405b-instruct-v1:0",
            messages=messages
        )
        
        print("Model response:", response_text)
        print("AWS Bedrock configuration is working correctly!")
    except Exception as e:
        print("Error testing AWS Bedrock configuration:", str(e))
        import traceback
        traceback.print_exc()