# Desafio MBA Engenharia de Software com IA - Full Cycle

Sistema de **ingestão e busca semântica** usando LangChain e PostgreSQL com pgVector: ingere um PDF em vetores e responde perguntas via CLI apenas com base no conteúdo do documento.

## Pré-requisitos

- Python 3.9+
- Docker e Docker Compose
- API Key da **OpenAI** ou da **Google** (Gemini)

## Configuração

1. Copie o template de variáveis de ambiente:

   ```bash
   cp .env.example .env
   ```

2. Edite o `.env` e preencha:
   - **OpenAI:** `OPENAI_API_KEY`
   - **ou Gemini:** `GOOGLE_API_KEY`
   - Ajuste `DATABASE_URL`, `PG_VECTOR_COLLECTION_NAME` e `PDF_PATH` se precisar (os valores padrão funcionam com o `docker-compose` e o `document.pdf` na raiz).

3. Crie e ative um ambiente virtual:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

   No Linux/macOS: `source venv/bin/activate`

4. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

## Ordem de execução

### 1. Subir o banco de dados

```bash
docker compose up -d
```

Aguarde o PostgreSQL com pgVector ficar saudável (cerca de 10–15 segundos).

### 2. Executar a ingestão do PDF

```bash
python src/ingest.py
```

Isso carrega o `document.pdf` (ou o caminho em `PDF_PATH`), divide em chunks de 1000 caracteres com overlap de 150, gera os embeddings e grava no PostgreSQL.

### 3. Rodar o chat

```bash
python src/chat.py
```

Digite as perguntas no terminal. O sistema usa somente o conteúdo do PDF. Para sair, digite `sair` ou use Ctrl+C.

## Exemplo no CLI

```
Faça sua pergunta: Qual o faturamento da Empresa SuperTechIABrazil?

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.
```

Perguntas fora do contexto do PDF:

```
Faça sua pergunta: Quantos clientes temos em 2024?

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

## Sobre os exemplos no prompt

No `search.py`, o prompt enviado à LLM inclui **exemplos de perguntas que podem ser respondidas** com base no documento (ex.: faturamento da SuperTechIABrazil, ano de fundação da Alfa IA Indústria) e exemplos de perguntas **fora do contexto**. Esses exemplos foram adicionados porque, nos testes, o modelo tendia a se recusar a responder todas as perguntas, inclusive as que tinham resposta explícita no contexto do PDF. Com os exemplos no prompt, o modelo passa a distinguir melhor entre perguntas válidas sobre o documento e perguntas que devem receber *"Não tenho informações necessárias para responder sua pergunta."*

## Tecnologias

- **Linguagem:** Python  
- **Framework:** LangChain  
- **Banco de dados:** PostgreSQL + pgVector  
- **Embeddings:** OpenAI (`text-embedding-3-small`) ou Google (`models/embedding-001`)  
- **LLM:** OpenAI (ex.: `gpt-4o-mini`) ou Google (ex.: `gemini-2.0-flash`)

## Estrutura do projeto

```
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── src/
│   ├── ingest.py    # Ingestão do PDF em vetores (pgVector)
│   ├── search.py    # Busca vetorial, montagem do prompt e chamada à LLM
│   ├── chat.py      # CLI para perguntas e respostas
├── document.pdf
└── README.md
```
