# Rxivonauta 🚀

Um sistema automatizado de pesquisa acadêmica que utiliza LLMs (Large Language Models) e a API do Arxiv para encontrar, analisar e selecionar artigos científicos relevantes.

## 🎯 Visão Geral

O Rxivonauta é um pipeline de pesquisa acadêmica que:
1. Gera queries otimizadas a partir de um tema de pesquisa
2. Busca artigos relevantes no Arxiv
3. Analisa e seleciona os artigos mais pertinentes
4. Produz um relatório detalhado dos resultados

### ✨ Características

- 🔍 Geração inteligente de queries usando LLMs
- 🌐 Suporte multilíngue (você pergunta em qualquer idioma, buscamos em inglês)
- 📊 Análise automática de relevância
- 📝 Relatórios detalhados
- ⚡ Processamento assíncrono
- 🔄 Rate limiting e retry automático

## 🛠️ Instalação

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

## ⚙️ Configuração

Edite o arquivo `.env` com suas configurações:

```env
# OpenRouter API
OPENROUTER_API_KEY=sua_chave_api_aqui
OPENROUTER_MODEL=google/gemini-2.0-pro-exp-02-05:free

# Logging
LOG_LEVEL=INFO
```

## 🚀 Uso

### Linha de Comando Básica

```bash
rxivonauta "seu tema de pesquisa"
```

### Opções Avançadas

```bash
rxivonauta "seu tema de pesquisa" [opções]

Opções disponíveis:
  --output-dir PATH           Diretório de saída
  --output-lang LANG         Idioma de saída (pt-BR, en-US, etc)
  --model MODEL              Modelo LLM a ser usado
  --max-queries N            Máximo de queries (default: 5)
  --max-results-per-query N  Resultados por query (default: 5)
  --max-age-days N          Idade máxima dos artigos (default: 365)
  --categories [CAT ...]     Categorias Arxiv (default: cs.AI cs.CL)
  --min-score FLOAT         Score mínimo (0-1, default: 0.6)
  --temperature FLOAT       Temperatura LLM (0-1, default: 0.7)
  --debug                   Ativa logs de debug
```

### Exemplos de Uso

1. Pesquisa básica:
```bash
rxivonauta "Machine Learning em Finanças"
```

2. Com configurações personalizadas:
```bash
rxivonauta "Redes Neurais Profundas" \
  --output-lang pt-BR \
  --model anthropic/claude-3.5-haiku \
  --max-queries 10 \
  --categories cs.AI cs.LG stat.ML \
  --min-score 0.7
```

3. Com saída personalizada:
```bash
rxivonauta "Análise de Séries Temporais" \
  --output-dir ./minha_pesquisa \
  --debug
```

### Testando o Pacote

Para testar o pacote, siga estas etapas:

1. Certifique-se de que você tem Python 3.8 ou superior instalado em seu sistema.
2. Instale as dependências necessárias executando `pip install -r requirements.txt` a partir do diretório raiz do repositório.
3. Verifique se o arquivo `.env` está configurado corretamente com as variáveis de ambiente necessárias, especialmente a `OPENROUTER_API_KEY`.
4. Execute o script principal com um tema de pesquisa de exemplo para testar todo o pipeline. Por exemplo, execute `python main.py "Artificial Intelligence" --output-lang en-US` a partir do diretório raiz.
5. Verifique os arquivos de saída gerados no diretório `data/processed` para garantir que o pipeline foi executado corretamente.
6. Revise os logs no arquivo `logs/rxivonauta.log` para verificar se há erros ou avisos durante a execução.

## 📊 Exemplo de Saída

```
Rxivonauta - Resumo da Execução
==================================================
Tema: Deep Learning for Time Series
Idioma: 🇧🇷 Português (Brasil)
Total de artigos encontrados: 25
Artigos selecionados: 8
Queries geradas: 5
Tempo de processamento: 45.82 segundos
Arquivos gerados:
  - Raw: raw_articles_20240214_235425.csv
  - Processado: processed_articles_20240214_235444.csv
  - Revisão: literature_review_20240214_235513.md
==================================================
```

## 🔧 Modelos LLM Suportados

### Modelos Gratuitos
- `google/gemini-2.0-pro-exp-02-05:free` (default)
- `mistralai/mistral-7b-instruct:free`
- `microsoft/phi-3-medium-128k-instruct:free`

### Modelos Premium
- `anthropic/claude-3.5-sonnet`
- `openai/gpt-4o-mini`
- `perplexity/sonar-small-chat`

## 🌱 Desenvolvimento

### Estrutura do Projeto

```
rxivonauta/
├── config/
│   ├── settings.py          # Configurações
│   └── prompts.py          # Templates LLM
├── src/
│   ├── agents/             # Agentes principais
│   ├── utils/              # Utilitários
│   └── main.py            # Script principal
└── data/                  # Dados e resultados
```

### Testes

```bash
# Instalar dependências de desenvolvimento
pip install -e ".[dev]"

# Rodar testes
pytest tests/
```

## 🤝 Contribuindo

1. Fork o projeto
2. Crie sua branch (`git checkout -b feature/MinhaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/MinhaFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 📫 Contato

José Lopes - [evandeilton@gmail.com](mailto:evandeilton@gmail.com)

Projeto: [https://github.com/evandeilton/rxivonauta](https://github.com/evandeilton/rxivonauta)

## 🙏 Agradecimentos

- OpenRouter pela API
- ArXiv pela API de pesquisa
- Contribuidores open source
