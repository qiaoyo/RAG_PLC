# pip3 install transformers
# python3 deepseek_tokenizer.py
import transformers
import json
import re
import os


# This function is from /Users/bytedance/python_proj/PLC_RAG/split_json.py
def read_json_with_control_chars(file_path):
    """Reads a JSON file, cleaning up control characters before parsing."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # This regex part is to handle potential errors in JSON format
            # It replaces unescaped newlines and tabs, and removes other control characters.
            cleaned_content = re.sub(r'(?<!\\)\\n', r'\\n', content)
            cleaned_content = re.sub(r'(?<!\\)\\t', r'\\t', cleaned_content)
            cleaned_content = re.sub(r'[\x00-\x1F\x7F]', '', cleaned_content)
            
            data = json.loads(cleaned_content)
            return data
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Error location: Line {e.lineno}, Column {e.colno}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def calculate_json_token_count(json_file_path, tokenizer_dir):
    """
    Reads a JSON file, and calculates the total number of tokens for its content.
    
    Args:
        json_file_path (str): The path to the input JSON file.
        tokenizer_dir (str): The directory containing the tokenizer files.
        
    Returns:
        int: The total number of tokens, or None if an error occurs.
    """
    print("Loading tokenizer...")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_dir, trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None
    
    print(f"Reading and processing JSON file: {json_file_path}")
    data = read_json_with_control_chars(json_file_path)
    
    if data is None:
        print("Failed to read JSON data.")
        return None
        
    # Convert the entire JSON object back to a string to count all tokens
    content_string = json.dumps(data, ensure_ascii=False)
    
    print("Encoding content to calculate token count...")
    tokens = tokenizer.encode(content_string)
    
    token_count = len(tokens)
    
    return token_count

if __name__ == "__main__":
    # Example of original usage:
    # tokenizer = transformers.AutoTokenizer.from_pretrained("./", trust_remote_code=True)
    # result = tokenizer.encode("Hello!")
    # print(f"'Hello!' tokenized: {result}")

    # New functionality: Calculate token count for a JSON file
    # Using relative paths for portability
    tokenizer_path = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(os.path.dirname(os.path.dirname(tokenizer_path)), 'data', 'chunks', 'chunk_001.json')
    
    if not os.path.isfile(json_path):
        print(f"JSON file not found at: {json_path}")
    else:
        print(f"Calculating token count for {json_path}...")
        total_tokens = calculate_json_token_count(json_path, tokenizer_path)
        if total_tokens is not None:
            print(f"Total token count for the file is: {total_tokens}")