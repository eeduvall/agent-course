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
def invoke_bedrock_directly(client, model_id, messages, temperature=0.2, max_tokens=5000, tools=None):
    """
    Makes a direct API call to AWS Bedrock without using the LangChain wrapper.
    
    Args:
        client: The boto3 Bedrock client
        model_id: The Bedrock model ID to use
        messages: List of message objects
        temperature: Temperature for generation
        max_tokens: Maximum number of tokens to generate
        tools: List of tools available for the model to use
    """
    # Build the system prompt with tool descriptions
    system_parts = []
    system_message = None
    
    # Extract system message if present
    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_message = msg.content
            break
    
    if system_message:
        system_parts.append(system_message)
    
    # Add tool descriptions to system prompt if tools are provided
    if tools:
        tools_section = ["\n\nAVAILABLE TOOLS:", "When you need to use a tool, use the following format:",
                        "<tool_call>", "name=<tool_name>", "args={\"arg1\": \"value1\", \"arg2\": \"value2\"}", "</tool_call>",
                        "The available tools are:"]
        
        for tool in tools:
            try:
                if hasattr(tool, 'name') and hasattr(tool, 'description'):
                    tool_name = tool.name
                    tool_desc = tool.description
                elif callable(tool):
                    tool_name = getattr(tool, '__name__', 'tool')
                    tool_desc = getattr(tool, '__doc__', f'A function called {tool_name}')
                else:
                    continue
                    
                tools_section.append(f"- {tool_name}: {tool_desc}")
            except Exception as e:
                print(f"Error processing tool {tool}: {e}")
                continue
        
        system_parts.append("\n".join(tools_section))
    
    # Build conversation in the format Llama expects
    conversation = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue  # Already handled in system_parts
        elif isinstance(msg, HumanMessage):
            conversation.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                # Handle tool calls in the assistant's message
                for tool_call in msg.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    tool_call_str = f"<tool_call>\nname={tool_name}\nargs={tool_args}\n</tool_call>"
                    conversation.append({"role": "assistant", "content": tool_call_str})
            else:
                conversation.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            # Format tool messages with the result
            conversation.append({"role": "user", "content": f"<tool_result>\n{msg.content}\n</tool_result>"})
    
    # Combine system parts with the conversation
    full_system_prompt = "\n\n".join(system_parts)
    
    # Format the final prompt
    prompt_parts = [full_system_prompt]
    for msg in conversation:
        if msg["role"] == "user":
            prompt_parts.append(f"User: {msg['content']}")
        else:
            prompt_parts.append(f"Assistant: {msg['content']}")
    
    # Add assistant's turn
    prompt_parts.append("Assistant:")
    
    # Join all parts to form the final prompt
    final_prompt = "\n\n".join(prompt_parts)
    
    # Prepare the request body for Llama model
    request_body = {
        "prompt": final_prompt,
        "temperature": temperature,
        "max_gen_len": max_tokens
    }
    
    # The prompt is already fully constructed, no need to reformat it here
    # Just ensure it's properly stripped
    request_body["prompt"] = final_prompt.strip()
    
    # Convert to JSON string
    body = json.dumps(request_body)

    success = False
    attempts = 0

    while not success and attempts < 2:
        # Make the API call
        response = client.invoke_model(
            modelId=model_id,
            body=body
        )
        
        # Parse the response
        response_body = json.loads(response.get('body').read())
        
        # Check if the response is valid
        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            success = True
        else:
            print(f"Invalid response format: {response_body}")
            attempts += 1
    
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
        
        # Log tools if provided
        if tools:
            print(f"--- Note: {len(tools)} tools provided ---")
            for tool in tools:
                print(f"Tool: {tool.name} - {tool.description}")
        
        # Make direct API call to Bedrock with tools
        try:
            response_text = invoke_bedrock_directly(
                client=client,
                model_id=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools
            )
            
            print(f"--- Received response from AWS Bedrock ---")
            print(f"Response: {response_text[:500]}...")  # Show more of the response for debugging
            
            # Check if the response contains tool calls
            if "<tool_call>" in response_text and "</tool_call>" in response_text:
                # Extract tool call information
                import re
                tool_call_matches = re.findall(
                    r'<tool_call>(.*?)</tool_call>', 
                    response_text, 
                    re.DOTALL
                )
                
                tool_calls = []
                for match in tool_call_matches:
                    # Parse the tool call (this is a simplified example)
                    # You might need to adjust this based on your model's output format
                    tool_name_match = re.search(r'name=(.*?)\n', match)
                    tool_args_match = re.search(r'args=(\{.*?\})', match, re.DOTALL)
                    
                    if tool_name_match and tool_args_match:
                        tool_name = tool_name_match.group(1)
                        try:
                            tool_args = eval(tool_args_match.group(1))  # Be careful with eval in production
                            tool_calls.append({
                                'name': tool_name,
                                'args': tool_args
                            })
                        except:
                            print(f"Failed to parse tool args: {tool_args_match.group(1)}")
                
                if tool_calls:
                    # Create an AIMessage with tool calls
                    return AIMessage(
                        content=response_text,
                        tool_calls=[
                            {
                                'name': call['name'],
                                'args': call['args'],
                                'id': f"call_{i}"  # Generate a unique ID for each tool call
                            }
                            for i, call in enumerate(tool_calls)
                        ]
                    )
            
            # If no tool calls, return a regular AIMessage
            return AIMessage(content=response_text)
            
        except Exception as api_error:
            print(f"Error in direct Bedrock API call: {str(api_error)}")
            import traceback
            traceback.print_exc()
            raise api_error
        
    except Exception as e:
        print(f"Error invoking AWS Bedrock: {str(e)}")
        import traceback
        traceback.print_exc()
        return AIMessage(content=f"Error: AWS Bedrock call failed: {str(e)}")
