import logging
import json
import time
from typing import Dict, Optional, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential

from openai import OpenAI, APIError, APIConnectionError, RateLimitError

from config import settings

logger = logging.getLogger(__name__)

class OpenRouterClient:
    """
    Cliente assíncrono para a API do OpenRouter usando o SDK da OpenAI.
    """
    DEFAULT_MODEL = "google/gemini-2.0-flash-001"
    MODELS = [
        "google/gemini-2.0-flash-001",
        "google/gemini-2.0-flash-thinking-exp:free",
        "google/gemini-2.0-flash-lite-preview-02-05:free",
        "google/gemini-2.0-pro-exp-02-05:free",
        "mistralai/mistral-7b-instruct:free",
        "mistralai/mistral-small-24b-instruct-2501:free",
        "anthropic/claude-3.5-haiku-20241022:beta",
        "openai/o3-mini",
        "openai/o3-mini-high",
        "perplexity/llama-3.1-sonar-small-128k-chat",
        "microsoft/phi-3-medium-128k-instruct:free",
        "meta-llama/llama-3.3-70b-instruct:free"
    ]

    def __init__(
        self,
        api_key: str = settings.OpenRouterConfig.API_KEY,
        base_url: str = settings.OpenRouterConfig.BASE_URL,
        model: Optional[str] = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None
    ):
        """
        Inicializa o cliente da API.

        Args:
            api_key: Chave da API do OpenRouter
            base_url: URL base da API
            model: Modelo LLM a ser usado (opcional)
            site_url: URL opcional do site
            site_name: Nome opcional do site
        """
        if not api_key:
            raise ValueError("OpenRouter API key is missing. Please check your .env file.")
        
        self.api_key = api_key
        self.base_url = base_url
        
        # If no model specified, use the one from settings or default
        self.model = model or settings.OpenRouterConfig.MODEL
        if self.model not in self.MODELS:
            self.model = self.DEFAULT_MODEL
            logger.warning(f"Model {model} not found in available models. Using default: {self.DEFAULT_MODEL}")
        
        self.headers = {
            "HTTP-Referer": site_url or settings.OpenRouterConfig.SITE_URL,
            "X-Title": site_name or settings.OpenRouterConfig.SITE_NAME
        }
        
        self.client = self._create_client()
        logger.info("OpenRouterClient initialized successfully")

    def _create_client(self) -> OpenAI:
        """
        Cria um novo cliente OpenAI.

        Returns:
            OpenAI: Cliente configurado
        """
        logger.info("Creating OpenAI client")
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=self.headers
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry_error_callback=lambda retry_state: retry_state.outcome.result()
    )
    def generate_text(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 5000,
        max_retries: int = 3
    ) -> str:
        """
        Gera texto usando o modelo LLM via OpenRouter com suporte a múltiplos modelos.

        Args:
            prompt: Texto do prompt
            system_message: Mensagem de sistema opcional
            temperature: Temperatura para geração (criatividade)
            max_tokens: Número máximo de tokens na resposta
            max_retries: Número máximo de tentativas com modelos diferentes

        Returns:
            Texto gerado pelo modelo

        Raises:
            ValueError: Se houver erro na geração do texto após todas as tentativas
        """
        logger.info("Starting text generation")
        
        # Cria uma cópia mutável dos modelos para evitar modificar a lista original
        available_models = self.MODELS.copy()
        
        # Remove o modelo atual do início da lista para tentar outros primeiro
        if self.model in available_models:
            available_models.remove(self.model)
        available_models.append(self.model)

        last_error = None
        
        for attempt, model in enumerate(available_models, 1):
            try:
                logger.info(f"Attempt {attempt}/{len(available_models)}: Using model {model}")
                
                messages = [
                    {"role": "user", "content": prompt}
                ]
                if system_message:
                    messages.insert(0, {"role": "system", "content": system_message})

                logger.info(f"Sending request - Model: {model}, Prompt length: {len(prompt)}")
                
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=max(0.0, min(1.0, temperature)),
                    max_tokens=max_tokens
                )
                
                if not completion or not completion.choices:
                    raise ValueError(f"Empty response from model {model}")

                generated_text = completion.choices[0].message.content
                logger.info(f"Successfully generated response with model {model}")
                return generated_text
            
            except (APIError, APIConnectionError, RateLimitError) as e:
                logger.warning(f"API error with model {model}: {str(e)}")
                last_error = e
                if isinstance(e, RateLimitError):
                    time.sleep(5)  # Espera 5 segundos antes de tentar novamente
                continue
            except Exception as e:
                logger.error(f"Unexpected error with model {model}: {str(e)}")
                last_error = e
                continue

        error_msg = f"Failed to generate text after {len(available_models)} attempts. Last error: {last_error}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    def close(self):
        """Fecha a sessão do cliente (não necessário com o SDK da OpenAI)."""
        pass