"""
Ollama API client for interacting with Gemma models.
"""

import requests
import json
from typing import List, Dict, Optional

class OllamaClient:
    """Client for Ollama API."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama API base URL (default: http://localhost:11434)
        """
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
    
    def check_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False
    ) -> str:
        """
        Generate text using Ollama model.
        
        Args:
            model: Model name (e.g., 'gemma2:2b' or 'gemma3:1b')
            prompt: Input prompt
            system: System message (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
        
        Returns:
            Generated text
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        if system:
            payload["system"] = system
        
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                if stream:
                    # Handle streaming response
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            if 'response' in data:
                                full_response += data['response']
                            if data.get('done', False):
                                break
                    return full_response
                else:
                    data = response.json()
                    return data.get('response', '')
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return ""
        except Exception as e:
            print(f"Error generating text: {e}")
            return ""
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """
        Chat with Ollama model.
        
        Args:
            model: Model name
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            Model response
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/chat",
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('message', {}).get('content', '')
            else:
                print(f"Error: {response.status_code} - {response.text}")
                return ""
        except Exception as e:
            print(f"Error in chat: {e}")
            return ""
    
    def pull_model(self, model: str) -> bool:
        """
        Pull a model from Ollama registry.
        
        Args:
            model: Model name to pull
        
        Returns:
            True if successful
        """
        try:
            response = requests.post(
                f"{self.api_url}/pull",
                json={"name": model},
                timeout=600,
                stream=True
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if 'status' in data:
                            print(data['status'])
                        if data.get('completed', False):
                            return True
                return True
            return False
        except Exception as e:
            print(f"Error pulling model: {e}")
            return False

def test_ollama():
    """Test Ollama connection and model."""
    client = OllamaClient()
    
    print("Checking Ollama connection...")
    if not client.check_connection():
        print("ERROR: Cannot connect to Ollama. Make sure Ollama is running on port 11434")
        return False
    
    print("✓ Ollama is running")
    
    print("\nAvailable models:")
    models = client.list_models()
    for model in models:
        print(f"  - {model}")
    
    if not models:
        print("No models found. Pull a model first:")
        print("  ollama pull gemma3:1b")
        return False
    
    # Test generation
    print("\nTesting generation with gemma3:1b...")
    test_prompt = "What is 2 + 2? Answer briefly."
    response = client.generate("gemma3:1b", test_prompt, temperature=0.7, max_tokens=50)
    
    print(f"Prompt: {test_prompt}")
    print(f"Response: {response}")
    
    return True

if __name__ == "__main__":
    test_ollama()

