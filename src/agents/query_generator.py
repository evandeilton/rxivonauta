# Agente para geração de queries de pesquisa
from dataclasses import dataclass
import json
import logging
from typing import Dict, List, Optional

from config import settings, prompts
from ..utils.api_client import OpenRouterClient

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Estrutura para armazenar resultados da geração de queries."""
    original_topic: str
    english_topic: str
    key_terms: List[str]
    queries: List[Dict]

class QueryGenerator:
    """
    Agente responsável por traduzir o tema de pesquisa e gerar queries otimizadas para o Arxiv.
    """
    
    def __init__(self, api_key: str = settings.OpenRouterConfig.API_KEY, temperature: float = 0.7):
        """
        Inicializa o gerador de queries.
        
        Args:
            api_key: Chave da API do OpenRouter
            temperature: Temperatura para geração de texto
        """
        if not api_key:
            raise ValueError("OpenRouter API key is required")
            
        self.client = OpenRouterClient(api_key)
        self.temperature = min(max(temperature, 0), 1)  # Ensure between 0 and 1

        self._state = {
            'original_topic': None,
            'english_topic': None,
            'translation_data': None
        }

    def translate_topic(self, research_topic: str) -> Dict:
        """
        Traduz o tema de pesquisa para inglês se necessário.
        
        Args:
            research_topic: Tema de pesquisa original
            
        Returns:
            Dict contendo:
                - translated_topic: Tema traduzido
                - key_terms: Lista de termos-chave
                - explanation: Explicação da tradução
                
        Raises:
            ValueError: Se a tradução falhar ou retornar formato inválido
        """
        if not research_topic or not research_topic.strip():
            raise ValueError("Research topic cannot be empty")
            
        try:
            # Gera o prompt de tradução
            prompt = prompts.format_translation_prompt(research_topic)
            
            # Obtém a tradução do LLM
            response = self.client.generate_text(
                prompt=prompt,
                system_message=prompts.SystemMessages.TRANSLATOR
            )
            
            logger.debug(f"Raw translation response: {response}")
            
            # Processa a resposta JSON
            translation_data = self._parse_json_response(
                response,
                required_fields=['translated_topic']
            )
            
            # Atualiza o estado
            self._state.update({
                'original_topic': research_topic,
                'english_topic': translation_data['translated_topic'],
                'translation_data': translation_data
            })
            
            return translation_data
            
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            raise ValueError(prompts.ErrorMessages.TRANSLATION_ERROR['pt']) from e

    def generate_queries(
        self,
        research_topic: str,
        num_queries: int = settings.SystemConfig.MAX_QUERIES
    ) -> QueryResult:
        """
        Gera queries de busca baseadas no tema de pesquisa.

        Args:
            research_topic: Tema de pesquisa
            num_queries: Número de queries a serem geradas

        Returns:
            QueryResult contendo tema original, traduzido e queries geradas

        Raises:
            ValueError: Se houver erro na geração das queries
        """
        try:
            logger.info(f"Generating {num_queries} queries for topic: {research_topic}")

            # Traduz o tema
            translation = self.translate_topic(research_topic)
            english_topic = translation['translated_topic']

            # Gera o prompt para queries
            prompt = prompts.format_query_prompt(
                original_topic=research_topic,
                english_topic=english_topic
            )

            # Obtém as queries do LLM
            response = self.client.generate_text(
                prompt=prompt,
                system_message=prompts.SystemMessages.QUERY_GENERATOR
            )

            # Processa a resposta JSON
            query_data = self._parse_json_response(response, required_fields=[])

            # Extrai as queries da resposta
            try:
                queries = query_data.get('queries', [])
                if not isinstance(queries, list):
                    raise ValueError("Queries must be a list")
            except (AttributeError, KeyError, TypeError) as e:
                logger.error(f"Error extracting queries: {str(e)}")
                raise ValueError("Invalid query response format") from e

            # Valida a estrutura das queries
            if not self._validate_query_response({"queries": queries}):
                raise ValueError("Invalid query response structure")

            # Limita o número de queries se necessário
            queries = queries[:num_queries]

            # Gera o resultado
            result = QueryResult(
                original_topic=research_topic,
                english_topic=english_topic,
                key_terms=translation.get('key_terms', []),
                queries=queries
            )

            logger.info(f"Successfully generated {len(queries)} queries")
            return result

        except Exception as e:
            logger.error(f"Query generation error: {str(e)}")
            raise

    def _parse_json_response(
        self,
        response: str,
        required_fields: List[str]
    ) -> Dict:
        """
        Processa e valida uma resposta JSON do LLM.
        
        Args:
            response: Resposta do LLM
            required_fields: Lista de campos obrigatórios
            
        Returns:
            Dict com os dados processados
            
        Raises:
            ValueError: Se o JSON for inválido ou campos obrigatórios ausentes
        """
        try:
            # Remove código markdown se presente
            clean_response = response.strip()
            if clean_response.startswith('```json'):
                clean_response = clean_response[7:]
            if clean_response.endswith('```'):
                clean_response = clean_response[:-3]
            
            # Parse do JSON
            data = json.loads(clean_response.strip())

            return data

            # # Tenta acessar 'queries' diretamente, senão busca em níveis mais profundos
            # if 'queries' not in data and isinstance(data, dict):
            #     for key, value in data.items():
            #         if isinstance(value, dict) and 'queries' in value:
            #             data = value
            #             #break

            # # Verifica campos obrigatórios
            # #missing_fields = [
            # #    field for field in required_fields
            # #    if field not in data
            # ]
            
            #if missing_fields:
            #    raise ValueError(f"Missing required fields: {missing_fields}")
                
            return data

        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {str(e)}")
            logger.error(f"Raw response: {response}")
            return {"queries": []}

    def _validate_query_response(self, response: Dict) -> bool:
        """
        Valida a estrutura da resposta de geração de queries.
        
        Args:
            response: Resposta processada da API
            
        Returns:
            bool indicando se a resposta é válida
        """
        if not isinstance(response, dict) or 'queries' not in response:
            return False
            
        queries = response['queries']
        if not isinstance(queries, list) or not queries:
            return False
            
        required_fields = {'query_id', 'query_text', 'rationale', 'aspect'}
        
        for query in queries:
            if not isinstance(query, dict):
                return False
                
            if not all(field in query for field in required_fields):
                return False
                
            # Valida tipos dos campos
            if not isinstance(query['query_id'], str):
                return False
            if not isinstance(query['query_text'], str):
                return False
            if not isinstance(query['rationale'], str):
                return False
            if not isinstance(query['aspect'], str):
                return False
                
        return True
    
    def get_topics(self) -> Optional[Dict[str, str]]:
        """
        Retorna os temas original e traduzido do último processamento.
        
        Returns:
            Dict contendo os temas ou None se não houver tradução
        """
        if self._state['original_topic'] and self._state['english_topic']:
            return {
                'original': self._state['original_topic'],
                'english': self._state['english_topic']
            }
        return None
