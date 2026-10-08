# Contador de Parafusos

Microsserviço FastAPI para processar fotos de parafusos com o pipeline do notebook `contar_parafuso_batch.ipynb`.

O serviço recebe uma imagem, executa o pré-processamento, aplica morfologia e watershed, filtra os contornos por métricas de forma e devolve uma imagem final com:

- máscara binária
- contornos válidos destacados
- contagem total de parafusos
- marcação dos itens rejeitados

## Estrutura

- `app.py`: API FastAPI e interface web básica
- `processing.py`: pipeline de visão computacional
- `templates/index.html`: página de upload
- `static/styles.css`: estilo da interface
- `requirements.txt`: dependências Python
- `Dockerfile`: imagem de produção

## Executar localmente

1. Crie um ambiente virtual.
2. Instale as dependências.
3. Inicie o servidor.

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Abra `http://127.0.0.1:8000`.

## Endpoint

### `POST /api/process`

Recebe `multipart/form-data` com o campo `file`.

Resposta:

- `image/png` com a imagem processada
- header `X-Screw-Count` com a contagem
- header `X-Rejected-Count` com a quantidade rejeitada

## Docker

```bash
docker build -t contador-parafusos .
docker run -p 8000:8000 contador-parafusos
```

## Ajustes do pipeline

Os principais parâmetros estão em `processing.py`:

- `BLUR_KERNEL`
- `CLOSE_KERNEL`
- `MORPH_KERNEL`
- `AREA_MIN` / `AREA_MAX`
- `SOLIDEZ_MIN`
- `ASPECTO_MIN` / `ASPECTO_MAX`
- `EXTENT_MIN`
- `USAR_WATERSHED`

## Observação

As imagens de exemplo do repositório podem ser usadas para teste local, mas o diretório raiz ignora novos arquivos de imagem gerados durante a experimentação.