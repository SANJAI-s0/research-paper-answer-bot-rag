"""
Gemini LLM Client module.
Provides robust integration with Google Gemini (gemini-2.5-flash / gemini-1.5-flash)
strictly conforming to project requirements (Zero OpenAI).
"""

from typing import Optional, Dict, Any
import os
from src.utils import get_logger
from src.config import config

logger = get_logger("GeminiClient")


class GeminiLLMClient:
    """Wrapper for Google Gemini API."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = config.GEMINI_TEMPERATURE,
    ):
        self.api_key = api_key or config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name or config.GEMINI_MODEL
        self.temperature = temperature
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the Gemini client using google-genai or google.generativeai."""
        if not self.api_key or self.api_key in ("your_gemini_api_key_here", "dummy_key"):
            logger.warning("GEMINI_API_KEY is not set or using placeholder. Live generation requires a valid key.")
            return

        try:
            # First attempt: modern google-genai SDK
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            self._sdk_type = "google-genai"
            logger.info(f"Initialized google-genai client with model '{self.model_name}'.")
        except Exception as e1:
            try:
                # Fallback attempt: google.generativeai
                import google.generativeai as genai_legacy
                genai_legacy.configure(api_key=self.api_key)
                self._client = genai_legacy.GenerativeModel(
                    model_name=self.model_name,
                    generation_config={"temperature": self.temperature},
                )
                self._sdk_type = "google-generativeai"
                logger.info(f"Initialized google.generativeai client with model '{self.model_name}'.")
            except Exception as e2:
                logger.error(f"Failed to initialize Gemini client: {str(e1)} | {str(e2)}")
                self._client = None

    def is_configured(self) -> bool:
        """Check if a valid Gemini client is ready."""
        return self._client is not None and bool(self.api_key)

    def generate_response(self, system_instruction: str, user_prompt: str) -> str:
        """Generate response from Gemini given system instructions and user prompt."""
        if not self.is_configured():
            # Try reinitializing if environment variable was loaded dynamically
            self.api_key = config.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
            self._init_client()
            
        if not self.is_configured():
            return (
                "[!] [Gemini API Key Required]: Please set your GEMINI_API_KEY in the .env file or Streamlit sidebar.\n\n"
                "Retrieved context passages were found and are displayed below in the 'Top 3 Supporting Sources' section."
            )

        try:
            if self._sdk_type == "google-genai":
                from google.genai import types
                import time
                
                # Active supported models on Google Gemini v1beta API
                candidate_models = [
                    self.model_name,
                    "gemini-3.7-flash",
                    "gemini-2.5-flash-lite",
                    "gemini-flash-latest",
                    "gemini-flash-lite-latest",
                    "gemma-4-26b-a4b-it",
                ]
                
                # Deduplicate while preserving order
                unique_models = []
                for m in candidate_models:
                    if m and m not in unique_models:
                        unique_models.append(m)

                last_err = None
                for model in unique_models:
                    for attempt in range(3):
                        try:
                            response = self._client.models.generate_content(
                                model=model,
                                contents=user_prompt,
                                config=types.GenerateContentConfig(
                                    system_instruction=system_instruction,
                                    temperature=self.temperature,
                                ),
                            )
                            if response and response.text:
                                return response.text.strip()
                        except Exception as err:
                            last_err = err
                            err_msg = str(err).lower()
                            if "503" in err_msg or "unavailable" in err_msg or "429" in err_msg or "quota" in err_msg:
                                time.sleep(1.5 * (attempt + 1))
                                continue
                            else:
                                break  # Move to next model if model not found or invalid
                                
                raise last_err or RuntimeError("No response from available Gemini models.")

            elif self._sdk_type == "google-generativeai":
                response = self._client.generate_content(
                    f"{system_instruction}\n\n{user_prompt}"
                )
                return response.text.strip()

        except Exception as e:
            logger.error(f"Gemini API generation error: {str(e)}")
            return f"⚠️ Error generating response from Gemini API: {str(e)}"
        
        return config.REFUSAL_RESPONSE


def get_gemini_client() -> GeminiLLMClient:
    """Factory to retrieve default Gemini client."""
    return GeminiLLMClient()
