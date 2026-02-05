import re, json, os, logging
from typing import Dict, Any, Optional
from .model import LLMBackend
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
# Implement Conext Caching (Maybe with Langgraph)
load_dotenv()

logger = logging.getLogger(__name__)

class GeminiBackend(LLMBackend):
    def __init__(self, hyperparams: Dict):
        self.model_name = hyperparams['model_name']
        self.temperature = hyperparams.get('temperature', 0.7)
        
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini backend")
        
        genai.configure(api_key=self.api_key)
        logger.info(f"Configured Gemini model {self.model_name}")

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
            )
        )
        logger.info("Initialized Gemini Model")
    
    def generate(self, prompt: str) -> str:
        logger.info("Getting response from Gemini")
        try:
            response = self.model.generate_content(prompt)
            logger.info("Received response from Gemini")
            logger.info(f"RAW Output: {response.text}")
            if response.text:
                response_text = response.text.strip()
                # check if given in ```json ``` format
                
                fenced = re.search(r"```json\s*([\s\S]*?)```", response_text, re.IGNORECASE)
                if fenced:
                    try:
                        obj = json.loads(fenced.group(1).strip())
                        json_str = json.dumps(obj)  # normalize
                        logger.info(f"LLM fenced Response: {json_str}")
                        return json_str
                    except json.JSONDecodeError:
                        logger.error("Invalid JSON inside ```json fenced block```; falling back.")
                
                logger.info(response_text)
                # Extract JSON from response if present
                json_match = re.search(r'\{.*\}', response_text)
                if json_match:
                    json_str = json_match.group(0)
                    # Validate JSON
                    json.loads(json_str)
                    logger.info(f"LLM Response: {json_str}")
                    return json_str
                else:
                    logger.error(f"No valid JSON found in response: {response_text}")
                    return '{"action": "Monitor", "reason": "No valid JSON found in response"}'
            else:
                logger.warning("Empty response from Gemini")
                return '{"action": "Monitor", "reason": "Empty response from Gemini"}'
                
        except Exception as e:
            logger.error(f"Error generating response from Gemini: {e}")
            return '{"action": "Monitor", "reason": "Error generating response from Gemini"}'