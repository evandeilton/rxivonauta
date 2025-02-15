# Rxivonauta 🚀

Um sistema automatizado de pesquisa acadêmica que utiliza LLMs (Large Language Models) e a API do Arxiv para encontrar, analisar e selecionar artigos científicos relevantes.

## Visão Geral

O Rxivonauta é um pipeline de pesquisa acadêmica que:
1. Gera queries otimizadas a partir de um tema de pesquisa
2. Busca artigos relevantes no Arxiv
3. Analisa e seleciona os artigos mais pertinentes
4. Produz um relatório detalhado dos resultados

### Características

- 🔍 Geração inteligente de queries usando LLMs
- 🌐 Suporte multilíngue (você pergunta em qualquer idioma, buscamos em inglês)
- 📊 Análise automática de relevância
- 📝 Relatórios detalhados
- ⚡ Processamento assíncrono
- 🔄 Rate limiting e retry automático

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/evandeilton/rxivonauta.git
cd rxivonauta
```

2. Crie e ative um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente:
```bash
cp .env.example .env
# Edite o arquivo .env com suas configurações
```

## Configuração

Edite o arquivo `.env` com suas configurações:

```env
# OpenRouter API
OPENROUTER_API_KEY=sua_chave_api_aqui
OPENROUTER_MODEL=anthropic/claude-3-opus-20240229

# Logging
LOG_LEVEL=INFO
```

## Uso

### Linha de Comando

```bash
python -m rxivonauta.src.main "seu tema de pesquisa"
```

Opções disponíveis:
- `--output-dir`: Diretório personalizado para saída
```bash
python -m rxivonauta.src.main "seu tema de pesquisa" --output-dir /caminho/personalizado
```

### Exemplo de Saída

```
Rxivonauta - Resumo da Execução
==================================================
Tema: GLM Regression Models
Idioma: 🇧🇷 Português (Brasil)
Total de artigos encontrados: 25
Artigos selecionados: 9
Queries geradas: 5
Tempo de processamento: 77.82 segundos
Arquivos gerados:
  - Raw: raw_articles_20250214_235425.csv
  - Processado: processed_articles_20250214_235444.csv
  - Revisão: literature_review_20250214_235513.md
==================================================
```

## Estrutura do Projeto

```
rxivonauta/
├── config/
│   ├── settings.py          # Configurações
│   └── prompts.py           # Templates LLM
├── src/
│   ├── agents/
│   │   ├── query_generator.py   # Gerador de queries
│   │   ├── arxiv_searcher.py    # Buscador Arxiv
│   │   └── content_analyzer.py   # Analisador
│   ├── utils/
│   │   ├── api_client.py        # Cliente API
│   │   └── data_processor.py    # Processador
│   └── main.py                  # Script principal
└── data/
    ├── raw/                 # Dados brutos
    └── processed/           # Resultados
```

## Customização

### Ajustando Parâmetros

Você pode modificar vários parâmetros no arquivo `config/settings.py`:
- Número de queries geradas
- Artigos por query
- Categorias do Arxiv
- Rate limits
- etc.

### Modificando Prompts

Os templates de prompts para os LLMs podem ser ajustados em `config/prompts.py`.

## Output

O sistema gera dois tipos de arquivos:

1. **Dados Brutos** (`data/raw/`):
   - Todos os artigos encontrados
   - Metadados completos
   - Queries utilizadas

2. **Dados Processados** (`data/processed/`):
   - Artigos selecionados
   - Scores de relevância
   - Análises e justificativas
   - Estatísticas do processamento

## Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Dependências Principais

- Python 3.8+
- aiohttp
- arxiv
- pandas
- python-dotenv
- tenacity

## Log de Alterações

### [0.1.0] - 2024-02-14
- Lançamento inicial
- Suporte multilíngue
- Pipeline básico de pesquisa
- Análise automática de relevância

## Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Contato

José Lopes - [evandeilton@gmail.com](mailto:evandeilton@gmail.com)

Link do Projeto: [https://github.com/evandeilton/rxivonauta](https://github.com/evandeilton/rxivonauta)