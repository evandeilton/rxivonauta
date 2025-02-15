# Agente para busca de artigos no Arxiv
import arxiv
import logging
import os
import time
from typing import Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from tenacity import retry, stop_after_attempt, wait_exponential
import pandas as pd
from pathlib import Path

from config import settings
from config.settings import ArxivConfig

logger = logging.getLogger(__name__)

@dataclass
class ArxivArticle:
    """
    Estrutura de dados para artigos do Arxiv.
    
    Attributes:
        title: Título do artigo
        authors: Lista de autores
        published: Data de publicação
        updated: Data de atualização
        summary: Resumo do artigo
        pdf_url: URL do PDF
        arxiv_id: Identificador único do Arxiv
        categories: Categorias do artigo
        doi: Digital Object Identifier (opcional)
        journal_ref: Referência do journal (opcional)
        comment: Comentários adicionais (opcional)
        query_id: Identificador da query que encontrou o artigo
    """
    title: str
    authors: List[str]
    published: datetime
    updated: datetime
    summary: str
    pdf_url: str
    arxiv_id: str
    categories: List[str]
    doi: Optional[str]
    journal_ref: Optional[str]
    comment: Optional[str]
    query_id: str

    def __post_init__(self):
        """Validação após inicialização."""
        if not self.title or not self.title.strip():
            raise ValueError("Title cannot be empty")
        if not self.authors:
            raise ValueError("Authors list cannot be empty")
        if not self.arxiv_id:
            raise ValueError("Arxiv ID cannot be empty")
        if not self.query_id:
            raise ValueError("Query ID cannot be empty")

class ArxivSearcher:
    """
    Agente responsável por realizar buscas no Arxiv e processar os resultados.
    """
    
    def __init__(
        self, 
        max_results: int = ArxivConfig.MAX_RESULTS,
        rate_limit: float = ArxivConfig.RATE_LIMIT,
        sort_by: str = ArxivConfig.SORT_BY,
        sort_order: str = ArxivConfig.SORT_ORDER,
        max_age_days: int = ArxivConfig.MAX_AGE_DAYS
    ):
        """
        Inicializa o buscador do Arxiv.
        
        Args:
            max_results: Número máximo de resultados por query
            rate_limit: Tempo em segundos entre requisições
            sort_by: Critério de ordenação
            sort_order: Ordem de classificação
            max_age_days: Idade máxima dos artigos em dias
        """
        self.max_results = max_results
        self.rate_limit = rate_limit
        self.sort_by = arxiv.SortCriterion.SubmittedDate
        self.sort_order = sort_order.lower()
        self.max_age_days = max_age_days
        self.client = arxiv.Client()
        
    def search_articles(self, queries: List[Dict]) -> List[ArxivArticle]:
        """
        Realiza buscas no Arxiv para cada query fornecida.
        
        Args:
            queries: Lista de dicionários contendo as queries com:
                    - query_id: Identificador único
                    - query_text: Texto da busca
                    
        Returns:
            Lista de artigos encontrados
            
        Raises:
            ValueError: Se as queries forem inválidas
        """
        if not queries:
            raise ValueError("Empty queries list")
            
        for query in queries:
            if not isinstance(query, dict):
                raise ValueError(f"Invalid query format: {query}")
            if 'query_id' not in query or 'query_text' not in query:
                raise ValueError(f"Missing required fields in query: {query}")
        
        all_articles = []
        
        for query in queries:
            try:
                logger.info(f"Searching Arxiv for query: {query['query_text']}")
                
                # Adicionar filtros à query
                search_query = self._build_search_query(query['query_text'])
                
                # Construir a consulta
                search = arxiv.Search(
                    query=search_query,
                    max_results=self.max_results,
                    sort_by=self.sort_by
                )
                
                # Realizar a busca com retry
                articles = self._execute_search_with_retry(search, query['query_id'])
                all_articles.extend(articles)
                
                logger.info(f"Found {len(articles)} articles for query {query['query_id']}")
                
                # Respeitar rate limit
                time.sleep(self.rate_limit)
                                
            except Exception as e:
                logger.error(f"Error searching Arxiv for query {query['query_id']}: {str(e)}")
                continue
                
        return all_articles
    
    def _build_search_query(self, base_query: str) -> str:
        """
        Constrói a query completa com filtros adicionais.
        
        Args:
            base_query: Query base fornecida
            
        Returns:
            Query completa com filtros
        """
        # Adicionar filtro de categorias
        categories_filter = " OR ".join(f"cat:{cat}" for cat in ArxivConfig.CATEGORIES)
        
        # Adicionar filtro de data se configurado
        date_filter = ""
        if self.max_age_days is not None:
            date_filter = f" AND submittedDate:[NOW-{self.max_age_days}DAYS TO NOW]"
        
        return f"({base_query}) AND ({categories_filter}){date_filter}"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _execute_search_with_retry(
        self,
        search: arxiv.Search,
        query_id: str
    ) -> List[ArxivArticle]:
        """
        Executa uma busca no Arxiv com retry automático.
        
        Args:
            search: Objeto de busca do Arxiv
            query_id: Identificador da query
            
        Returns:
            Lista de artigos processados
            
        Raises:
            Exception: Se a busca falhar após todas as tentativas
        """
        return self._execute_search(search, query_id)
    
    def _execute_search(
        self,
        search: arxiv.Search,
        query_id: str
    ) -> List[ArxivArticle]:
        """
        Executa uma busca no Arxiv e processa os resultados.
        
        Args:
            search: Objeto de busca do Arxiv
            query_id: Identificador da query
            
        Returns:
            Lista de artigos processados
        """
        articles = []
        
        try:
            for result in search.results():
                try:
                    # Validar resultado antes de processar
                    if not self._validate_result(result):
                        continue
                        
                    article = self._process_article(result, query_id)
                    articles.append(article)
                except Exception as e:
                    logger.error(f"Error processing article {result.entry_id}: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error executing Arxiv search: {str(e)}")
            raise
            
        return articles
    
    def _validate_result(self, result: arxiv.Result) -> bool:
        """
        Valida um resultado do Arxiv antes do processamento.
        
        Args:
            result: Resultado do Arxiv
            
        Returns:
            bool indicando se o resultado é válido
        """
        try:
            if not result.title or not result.title.strip():
                return False
            if not result.authors:
                return False
            if not result.entry_id:
                return False
            if not result.summary or not result.summary.strip():
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating result: {str(e)}")
            return False
    
    def _process_article(self, result: arxiv.Result, query_id: str) -> ArxivArticle:
        """
        Processa um resultado do Arxiv para o formato padronizado.
        
        Args:
            result: Resultado do Arxiv
            query_id: Identificador da query
            
        Returns:
            ArxivArticle contendo os dados processados
            
        Raises:
            ValueError: Se os dados do artigo forem inválidos
        """
        try:
            return ArxivArticle(
                title=result.title.strip(),
                authors=[author.name for author in result.authors],
                published=result.published,
                updated=result.updated,
                summary=result.summary.strip(),
                pdf_url=result.pdf_url,
                arxiv_id=result.entry_id.split('/')[-1],
                categories=result.categories,
                doi=result.doi,
                journal_ref=result.journal_ref,
                comment=result.comment,
                query_id=query_id
            )
        except Exception as e:
            logger.error(f"Error processing article data: {str(e)}")
            raise
            
    def save_to_raw(
        self,
        articles: List[ArxivArticle],
        filename: str,
        output_dir: Optional[Path] = None
    ):
        """
        Salva os artigos encontrados no diretório de dados raw.
        
        Args:
            articles: Lista de artigos a serem salvos
            filename: Nome do arquivo de saída
            output_dir: Diretório opcional para saída
            
        Raises:
            ValueError: Se a lista de artigos estiver vazia
        """
        if not articles:
            raise ValueError("Empty articles list")
            
        try:
            # Definir diretório de saída
            output_dir = output_dir or settings.Directories.RAW_DATA_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Converter artigos para dicionários
            articles_data = [asdict(article) for article in articles]
            
            # Criar DataFrame
            df = pd.DataFrame(articles_data)
            
            # Processar campos de data
            for col in ['published', 'updated']:
                df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
                
            # Processar campos de lista
            for col in ['authors', 'categories']:
                df[col] = df[col].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)
            
            # Salvar arquivo
            output_path = output_dir / filename
            df.to_csv(output_path, index=False)
            
            logger.info(f"Saved {len(articles)} articles to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving articles to file: {str(e)}")
            raise