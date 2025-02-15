import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from config import prompts, settings
from config.prompts import ArticleAnalysis
from src.utils.api_client import OpenRouterClient

logger = logging.getLogger(__name__)

class ContentAnalyzer:
    """
    Agente responsável por analisar a relevância dos artigos encontrados.
    """
    
    def __init__(
        self, 
        api_key: str = settings.OpenRouterConfig.API_KEY,
        max_per_query: int = settings.SystemConfig.MAX_SELECTED_PER_QUERY,
        batch_size: int = settings.SystemConfig.CHUNK_SIZE,
        min_score_threshold: float = 0.6
    ):
        """
        Inicializa o analisador de conteúdo.
        
        Args:
            api_key: Chave da API do OpenRouter
            max_per_query: Número máximo de artigos por query
            batch_size: Tamanho do lote para processamento
            min_score_threshold: Score mínimo para seleção
        """
        if not api_key:
            raise ValueError("OpenRouter API key is required")
            
        self.client = OpenRouterClient(api_key)
        self.max_per_query = max_per_query
        self.batch_size = batch_size
        self.min_score_threshold = min_score_threshold
        
    def analyze_articles(
        self, 
        articles_df: pd.DataFrame,
        original_topic: str,
        english_topic: str,
        queries: List[Dict]
    ) -> pd.DataFrame:
        """
        Analisa os artigos e seleciona os mais relevantes.
        
        Args:
            articles_df: DataFrame com os artigos do Arxiv
            original_topic: Tema de pesquisa original
            english_topic: Tema traduzido para inglês
            queries: Lista de queries utilizadas
            
        Returns:
            DataFrame com os artigos selecionados e suas análises
            
        Raises:
            ValueError: Se os dados de entrada forem inválidos
        """
        try:
            if articles_df.empty:
                raise ValueError("Empty articles DataFrame")
                
            if not original_topic or not english_topic:
                raise ValueError("Topics cannot be empty")
                
            if not queries:
                raise ValueError("Empty queries list")
                
            logger.info(f"Starting analysis of {len(articles_df)} articles")
            
            # Criar dicionário de queries
            query_dict = self._create_query_dict(queries)
            
            # Processar artigos em lotes por query
            all_analyses = []
            for query_id in articles_df['query_id'].unique():
                query_articles = articles_df[articles_df['query_id'] == query_id]
                query_text = query_dict.get(query_id)
                
                if not query_text:
                    logger.warning(f"Query text not found for query_id: {query_id}")
                    continue
                
                logger.info(f"Analyzing {len(query_articles)} articles for query {query_id}")
                
                # Processar em lotes
                for batch_start in range(0, len(query_articles), self.batch_size):
                    batch_end = min(batch_start + self.batch_size, len(query_articles))
                    batch = query_articles.iloc[batch_start:batch_end]
                    
                    analyses = self._analyze_articles_batch(
                        batch,
                        original_topic,
                        english_topic,
                        query_text
                    )
                    all_analyses.extend(analyses)
            
            if not all_analyses:
                raise ValueError("No valid analyses generated")
            
            # Converter análises para DataFrame
            analysis_df = pd.DataFrame([asdict(a) for a in all_analyses])
            
            # Selecionar melhores artigos
            selected_df = self._select_best_articles(analysis_df)
            
            # Mesclar resultados
            result_df = self._merge_analysis_results(articles_df, selected_df)
            
            logger.info(f"Analysis completed. Selected {len(result_df)} articles")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error in article analysis: {str(e)}")
            raise
            
    def _create_query_dict(self, queries: List[Dict]) -> Dict[str, str]:
        """
        Cria dicionário de queries para fácil acesso.
        
        Args:
            queries: Lista de queries
            
        Returns:
            Dicionário mapeando query_id para query_text
        """
        return {
            q['query_id']: q['query_text'] 
            for q in queries 
            if 'query_id' in q and 'query_text' in q
        }
            
    def _analyze_articles_batch(
        self,
        articles_batch: pd.DataFrame,
        original_topic: str,
        english_topic: str,
        query_text: str
    ) -> List[ArticleAnalysis]:
        """
        Analisa um lote de artigos em paralelo.
        
        Args:
            articles_batch: Lote de artigos para análise
            original_topic: Tema original
            english_topic: Tema em inglês
            query_text: Texto da query
            
        Returns:
            Lista de análises dos artigos
        """
        analyses = []
        
        with ThreadPoolExecutor() as executor:
            futures = []
            
            for _, article in articles_batch.iterrows():
                future = executor.submit(
                    self._analyze_single_article,
                    article,
                    original_topic,
                    english_topic,
                    query_text
                )
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        analyses.append(result)
                except Exception as e:
                    logger.error(f"Error in article analysis thread: {str(e)}")
                    
        return analyses
    
    def _analyze_single_article(
        self,
        article: pd.Series,
        original_topic: str,
        english_topic: str,
        query_text: str
    ) -> Optional[ArticleAnalysis]:
        """
        Analisa um único artigo.
        
        Args:
            article: Dados do artigo
            original_topic: Tema original
            english_topic: Tema em inglês
            query_text: Texto da query
            
        Returns:
            ArticleAnalysis ou None se a análise falhar
        """
        response = None
        try:
            # Preparar prompt
            prompt = prompts.format_analysis_prompt(
                article_data=article.to_dict(),
                original_topic=original_topic,
                english_topic=english_topic,
                query_text=query_text
            )

            # Obter análise do LLM
            response = self.client.generate_text(
                prompt=prompt,
                system_message=prompts.SystemMessages.CONTENT_ANALYZER
            )

            # Log the raw response for debugging
            logger.debug(f"Raw LLM response for article {article.get('arxiv_id', 'unknown')}: {response[:200]}...")
            
            # Processar resposta
            analysis_data = self._parse_llm_response(response)
            
            # Log the parsed data
            logger.debug(f"Parsed analysis data: {json.dumps(analysis_data, indent=2)}")
            
            # Criar objeto de análise com validação de tipos
            return ArticleAnalysis(
                arxiv_id=str(article['arxiv_id']),
                query_id=str(article['query_id']),
                relevance_score=float(analysis_data.get('relevance_score', 0.0)),
                selection_rationale=str(analysis_data.get('selection_rationale', '')),
                key_aspects=list(analysis_data.get('key_aspects', [])),
                methodology_score=float(analysis_data.get('methodology_score', 0.0)),
                impact_score=float(analysis_data.get('impact_score', 0.0)),
                language_notes=str(analysis_data.get('language_notes', ''))
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Error creating ArticleAnalysis object: {str(e)}")
            if response:
                logger.error(f"Response that caused error: {response[:200]}...")
            return None
        except Exception as e:
            logger.error(f"Error analyzing article {article.get('arxiv_id', 'unknown')}: {str(e)}")
            if response:
                logger.error(f"Response that caused error: {response[:200]}...")
            return None
    
    def _parse_llm_response(self, response: str) -> Dict:
        """
        Processa a resposta do LLM.
        
        Args:
            response: Resposta do modelo
            
        Returns:
            Dicionário com dados processados
            
        Raises:
            ValueError: Se a resposta for inválida
        """
        try:
            # Limpar resposta
            clean_response = response.strip()
            
            # Tentar encontrar o JSON na resposta
            start_idx = clean_response.find('{')
            end_idx = clean_response.rfind('}')
            
            if start_idx == -1 or end_idx == -1:
                raise ValueError("No JSON object found in response")
                
            json_str = clean_response[start_idx:end_idx + 1]
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Validar campos obrigatórios
            required_fields = [
                'relevance_score', 'selection_rationale', 'key_aspects',
                'methodology_score', 'impact_score', 'language_notes'
            ]
            
            missing = [f for f in required_fields if f not in data]
            if missing:
                logger.warning(f"Missing required fields: {missing}")
                for field in missing:
                    data[field] = "" if field == "selection_rationale" or field == "language_notes" else [] if field == "key_aspects" else 0.0

            # Ensure numeric fields are float
            for field in ['relevance_score', 'methodology_score', 'impact_score']:
                if field in data and not isinstance(data[field], float):
                    try:
                        data[field] = float(str(data[field]).strip())
                    except (ValueError, TypeError):
                        data[field] = 0.0

            return data
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Invalid JSON response: {str(e)}\nResponse: {response[:200]}...")
            return {
                "relevance_score": 0.0,
                "selection_rationale": "",
                "key_aspects": [],
                "methodology_score": 0.0,
                "impact_score": 0.0,
                "language_notes": ""
            }
    
    def _select_best_articles(self, analysis_df: pd.DataFrame) -> pd.DataFrame:
        """
        Seleciona os melhores artigos por query.
        
        Args:
            analysis_df: DataFrame com análises
            
        Returns:
            DataFrame com artigos selecionados
        """
        try:
            # Calcular scores ponderados
            weights = {
                'relevance_score': 0.5,      # 50%
                'methodology_score': 0.25,    # 25%
                'impact_score': 0.25         # 25%
            }
            
            analysis_df['final_score'] = sum(
                analysis_df[field] * weight
                for field, weight in weights.items()
            )
            
            # Bônus por aspectos-chave
            analysis_df['aspect_bonus'] = analysis_df['key_aspects'].apply(
                lambda x: min(len(x) * 0.05, 0.2) if isinstance(x, list) else 0
            )
            
            # Aplicar bônus
            analysis_df['final_score'] *= (1 + analysis_df['aspect_bonus'])
            
            # Normalizar scores
            max_score = analysis_df['final_score'].max()
            if max_score > 0:
                analysis_df['final_score'] /= max_score
            
            # Selecionar melhores por query
            selected = []
            for query_id, group in analysis_df.groupby('query_id'):
                qualified = group[group['final_score'] >= self.min_score_threshold]
                if not qualified.empty:
                    top_articles = qualified.nlargest(
                        self.max_per_query,
                        'final_score'
                    )
                    selected.extend(top_articles.to_dict('records'))
            
            selected_df = pd.DataFrame(selected)
            # Remove duplicate articles based on arxiv_id
            selected_df = selected_df.drop_duplicates(subset=['arxiv_id'])
            return selected_df
        except Exception as e:
            logger.error(f"Error selecting best articles: {str(e)}")
            raise
    
    def _merge_analysis_results(
        self,
        articles_df: pd.DataFrame,
        analysis_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combina dados originais com análises.
        
        Args:
            articles_df: DataFrame com artigos
            analysis_df: DataFrame com análises
            
        Returns:
            DataFrame combinado
        """
        try:
            result = pd.merge(
                articles_df,
                analysis_df,
                on=['arxiv_id', 'query_id'],
                how='inner',
                validate='1:1'
            )
            
            # Processar key_aspects
            result['key_aspects'] = result['key_aspects'].apply(
                lambda x: '; '.join(x) if isinstance(x, list) else x
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error merging analysis results: {str(e)}")
            raise
    
    def save_results(
        self,
        results_df: pd.DataFrame,
        filename: str,
        output_dir: Optional[Path] = None
    ):
        """
        Salva os resultados processados.
        
        Args:
            results_df: DataFrame com resultados
            filename: Nome do arquivo
            output_dir: Diretório opcional de saída
        """
        try:
            if results_df.empty:
                raise ValueError("Empty results DataFrame")
                
            output_dir = output_dir or settings.Directories.PROCESSED_DATA_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / filename
            results_df.to_csv(output_path, index=False)
            
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
