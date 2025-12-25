import os
import json
import yaml
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Data Conversion Utils ---

def json_to_compact_text(data: Dict, ignore_shp: List[str] = []) -> str:
    """
    Converts structured JSON to a compact text format for LLM context.
    """
    text_list = []
    if not isinstance(data, dict):
        return ""

    for shp_type, shp_values in data.items():
        if shp_type in ignore_shp or not isinstance(shp_values, list):
            continue
        
        shp_text = []
        for shp_value in shp_values:
            # Build text for a single feature
            item_parts = []
            for osm_id, info in shp_value.items():
                item_parts.append(f'osm_id: {osm_id}')
                
                if isinstance(info, dict):
                    for attr, val in info.items():
                        if val:
                            # Stringify lists/dicts to prevent format errors
                            val_str = json.dumps(val, ensure_ascii=False) if isinstance(val, (list, dict)) else str(val)
                            item_parts.append(f"{attr}:{val_str}")
            
            # Join attributes and remove newlines for compactness
            single_line = " ".join(item_parts).replace('\n', '')
            shp_text.append(single_line)
            
        # Add category header
        if shp_text:
            text_list.append(f"--- {shp_type} ---\n" + "\n".join(shp_text))
        
    return "\n\n".join(text_list)

# --- 2. LLM Client Wrapper ---

class UnifiedChatBot:
    def __init__(self, config: Dict):
        self.provider = config.get('model_provider', 'openai').lower()
        self.api_key = config.get('api_key')
        self.model_name = config.get('model_name')
        self.base_url = config.get('base_url')
        self.temperature = config.get('temperature', 0.1)
        self.client = None

        self._init_client()

    def _init_client(self):
        """Initialize specific SDK based on provider."""
        try:
            if self.provider == 'anthropic':
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
            elif self.provider == 'google':
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model_name)
            else:
                # Default to OpenAI (works for DeepSeek, Qwen, GPT)
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError as e:
            logging.error(f"Missing dependency for {self.provider}. Please install it.")
            raise e

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Unified chat interface. 
        messages format: [{'role': 'system'|'user'|'assistant', 'content': '...'}]
        """
        try:
            # 1. Google Gemini Handling
            if self.provider == 'google':
                # Gemini manages history differently (ChatSession), but here we simulate stateless for consistency
                # or we convert the `messages` list to Gemini's history format.
                # For simplicity in this wrapper, we construct a prompt or use chat history conversion.
                
                # Extract system prompt
                system_instruction = next((m['content'] for m in messages if m['role'] == 'system'), "")
                chat_history = []
                last_user_msg = ""

                for m in messages:
                    if m['role'] == 'user':
                        chat_history.append({"role": "user", "parts": [m['content']]})
                        last_user_msg = m['content'] # Keep track of last input
                    elif m['role'] == 'assistant':
                        chat_history.append({"role": "model", "parts": [m['content']]})
                
                # Remove the last user message from history as it will be sent in send_message
                if chat_history and chat_history[-1]['role'] == 'user':
                    chat_history.pop()

                # Re-init chat with system instruction (if supported by sdk version) or prepend it
                # Note: System instructions are model-dependent in Gemini. Here we prepend to context if needed.
                chat = self.client.start_chat(history=chat_history)
                
                # If system prompt exists and isn't supported natively, prepending it to the first message is a common fallback
                # But modern Gemini 1.5 supports system_instruction in constructor. 
                # For this generic script, we assume context is sufficient in history.
                response = chat.send_message(last_user_msg)
                return response.text

            # 2. Anthropic Claude Handling
            elif self.provider == 'anthropic':
                system_msg = next((m['content'] for m in messages if m['role'] == 'system'), "")
                user_assist_msgs = [m for m in messages if m['role'] != 'system']
                
                resp = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=2048,
                    temperature=self.temperature,
                    system=system_msg,
                    messages=user_assist_msgs
                )
                return resp.content[0].text

            # 3. OpenAI/DeepSeek/Qwen Handling
            else:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=self.temperature,
                    messages=messages
                )
                return resp.choices[0].message.content

        except Exception as e:
            logging.error(f"Inference Error: {e}")
            return f"Error generating response: {e}"

# --- 3. Main Logic ---

def main():
    # Load Configuration
    if not os.path.exists('config.yaml'):
        print("Error: config.yaml not found.")
        return
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Load Data
    data_path = config.get('input_file')
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return
    
    logging.info("Loading and converting data...")
    with open(data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    context_str = json_to_compact_text(raw_data)

    # Initialize System Prompt
    base_system_prompt = """You are a geospatial analysis assistant specializing in urban scenarios. 
Based on the data provided, perform spatial reasoning.
Data Schema includes: osmid, bbox, nodes, Topology, 3D Spatial, image_caption.
"""
    full_system_prompt = f"{base_system_prompt}\n\nDATA CONTEXT:\n{context_str}"

    # Initialize Chat History
    history = [{"role": "system", "content": full_system_prompt}]
    
    # Initialize Bot
    try:
        bot = UnifiedChatBot(config)
        logging.info(f"Model {config['model_name']} ({config['model_provider']}) initialized.")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return

    print("\n" + "="*50)
    print(" Geopatial Reasoning Assistant (Type 'exit' to quit)")
    print("="*50 + "\n")

    # Multi-turn Conversation Loop
    while True:
        try:
            user_input = input("\n[User]: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting...")
                break
            if not user_input:
                continue

            # Update history
            history.append({"role": "user", "content": user_input})

            # Inference
            print("Thinking...")
            response = bot.chat(history)

            # Output and update history
            print(f"\n[AI]: {response}")
            history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()