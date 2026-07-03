---
title: YOLOCraft
emoji: 🎮
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# YOLOCraft

Detecção e segmentação de mobs do Minecraft utilizando YOLO, SAM e visão computacional.

## Visão Geral

YOLOCraft identifica automaticamente mobs do Minecraft em imagens. Um modelo YOLO detecta a classe e a bounding box de cada mob; a máscara de segmentação é gerada via SAM (Segment Anything Model) ou por métodos clássicos de visão computacional (Otsu, HSV, GrabCut, Watershed), com um modo comparativo entre eles. Uma API expõe esse pipeline para uma aplicação web.

O projeto foi desenvolvido como estudo prático em:

* Computer Vision
* Object Detection
* Instance Segmentation
* Treinamento de modelos YOLO
* Organização de pipelines de Machine Learning

## Dataset

O dataset utilizado é:

* Minecraft Mobs
* Fonte: Kaggle
* 87 classes de mobs, mais de 27 mil imagens capturadas em jogo
* Anotações em CSV (bounding boxes normalizadas + metadados de cena como clima e distância), convertidas para o formato YOLO pelos scripts do projeto

Dataset:

https://www.kaggle.com/datasets/pierreayfri/minecraft-mobs/data

Um subconjunto de 16 classes (`data/minecraft_mobs-2/apresentacao`) foi curado com `src/data/dataset_manager.py` para o modelo de apresentação: cave_spider, creeper, enderman, skeleton, slime, spider, zombie, iron_golem, wolf, cat, chicken, cow, frog, horse, pig, sheep.

## Estado Atual

* Modelo de detecção (YOLO26s) treinado nas 16 classes curadas, 100 épocas: precision = 0.9906, recall = 0.9878, mAP50 = 0.9935, mAP50-95 = 0.9679.
* Segmentação em dois modos: SAM (MobileSAM, automático) e 4 métodos clássicos configuráveis (Otsu, HSV, GrabCut, Watershed), além de um modo "auto" que escolhe entre eles.
* API (`src/api.py`) completa e publicada, com cache de detecção por imagem (evita rodar o YOLO de novo entre os métodos de segmentação no modo comparativo).
* Deploy em produção no Hugging Face Spaces (Docker), integrado ao frontend.
* Histórico de treinos em `training_logs/training_history.csv`.

Pendências:

* Validação cruzada (k-fold, k=5) implementada em notebook (`notebooks/5_cross_validation/`), execução ainda pendente.

## Estrutura do Projeto

```text
YOLOCraft/
│
├── data/
│   ├── minecraft_mobs/          # dataset inicial (baseline, 4 classes)
│   └── minecraft_mobs-2/        # dataset principal (87 classes) + subconjuntos curados
│
├── notebooks/
│   ├── 1_exploracao/            # análise exploratória e visualização de labels
│   ├── 2_baseline/              # treinamento baseline
│   ├── 3_experimentos/          # experimentos de treinamento (usa TrainingLogger)
│   ├── 4_segmentation/          # testes de segmentação com SAM
│   ├── 5_cross_validation/      # validação cruzada (k-fold)
│   └── testes/                  # imagens de teste manual
│
├── src/
│   ├── api.py                   # API de inferência (FastAPI) — entrypoint do deploy
│   ├── config.py                # seleção de dataset (registro de paths)
│   ├── utils.py
│   ├── test_thresholds.py       # varredura de confidence threshold
│   │
│   ├── data/
│   │   ├── convert_dataset.py   # converte CSV de anotações para formato YOLO
│   │   └── dataset_manager.py   # app desktop (PyQt6) para curar o dataset
│   │
│   ├── training/
│   │   ├── training_logger.py   # registra histórico de treinos (JSON/CSV)
│   │   ├── train_with_logging.py
│   │   └── train_improved.py    # treino com hiperparâmetros de augmentation
│   │
│   ├── segmentation/
│   │   ├── segmentation.py      # pipeline de segmentação clássica (Otsu/HSV/GrabCut/Watershed)
│   │   ├── classic_segmentation.py
│   │   └── sam_segmentation.py  # wrapper de segmentação via SAM
│   │
│   └── gui/
│       ├── detector_gui.py      # app desktop (PyQt6) para testar modelos
│       └── detector_seg.py      # app desktop (PyQt6) comparando SAM x métodos clássicos
│
├── scripts/
│   └── download_dataset.py      # download automatizado via Kaggle CLI
│
├── docs/
│   └── frontend_integration.md  # contrato da API para o frontend
│
├── static/
│   └── samples/                 # imagens de teste servidas pela API (images/ + labels/ + data.yaml)
│
├── models/
│   └── production/              # modelo em produção (MOB_DET_YOLO_V1.pt, via Git LFS)
│
├── pretrained_models/           # pesos pré-treinados (YOLO, MobileSAM)
├── training_logs/               # histórico de treinos
├── requirements.txt
├── Dockerfile
├── README.md
└── .gitignore
```

## Instalação

Clone o repositório:

```bash
git clone https://github.com/seu-usuario/YOLOCraft.git
cd YOLOCraft
```

Crie um ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente:

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Configuração do Kaggle

O projeto utiliza a CLI oficial do Kaggle para baixar automaticamente o dataset.

Após instalar as dependências, execute:

```bash
kaggle auth login
```

Um link será exibido no terminal.

1. Abra o link no navegador;
2. Faça login na sua conta Kaggle;
3. Autorize o acesso;
4. Retorne ao terminal.

As credenciais serão armazenadas automaticamente e não será necessário repetir esse processo.

Para verificar se a autenticação foi realizada corretamente:

```bash
kaggle datasets list -s minecraft
```

## Download do Dataset

Após autenticar sua conta:

```bash
python scripts/download_dataset.py
```

O script irá:

* Verificar se o dataset já existe localmente;
* Fazer o download apenas quando necessário;
* Extrair os arquivos.

Os dados serão armazenados em:

```text
data/minecraft_mobs-2/
```

Converta as anotações (CSV) para o formato YOLO com `src/data/convert_dataset.py`, ou use `src/data/dataset_manager.py` para selecionar classes e exportar um subconjunto curado.

## Treinamento

O treinamento é feito nos notebooks de `notebooks/3_experimentos/`, que registram cada execução via `src/training/training_logger.py`. Também há um ponto de entrada em script:

```bash
python -m src.training.train_with_logging
```

Cada treino gera:

* Pesos do modelo
* Métricas de validação
* Curvas de aprendizado
* Um registro em `training_logs/`

Os resultados de cada execução ficam em:

```text
notebooks/3_experimentos/runs/
```

Validação cruzada (k-fold, k=5) disponível em `notebooks/5_cross_validation/`.

## Inferência e API

Duas ferramentas desktop para testar localmente:

* **`src/gui/detector_gui.py`** — carrega um modelo `.pt` e testa imagens com ajuste de confidence threshold.
* **`src/gui/detector_seg.py`** — compara lado a lado a segmentação via SAM e os métodos clássicos.

API (`src/api.py`):

```bash
uvicorn src.api:app --reload --port 8000
```

| Endpoint | Método | Descrição |
| --- | --- | --- |
| `/` | GET | status básico da API |
| `/health`, `/ping` | GET | checagem de disponibilidade |
| `/samples/classes` | GET | mobs disponíveis nas imagens de teste |
| `/samples` | GET | imagens de teste aleatórias, filtráveis por mob (`?mob=creeper&count=4`) |
| `/predict` | POST | detecção (YOLO) + segmentação via SAM |
| `/predict/classic` | POST | detecção (YOLO) + segmentação clássica (`?method=otsu\|hsv\|grabcut\|watershed\|auto`, parâmetros ajustáveis) |
| `/predict/classic/visualize` | POST | mesma resposta de `/predict/classic`, mas devolve um PNG anotado |

Contrato completo (formato de resposta e parâmetros) em `docs/frontend_integration.md`.

## Objetivos

### Detecção de Objetos

* Detectar mobs automaticamente em imagens utilizando modelos da família YOLO;
* Avaliar métricas como Precision, Recall, mAP50 e mAP50-95;
* Pipeline reproduzível de treinamento, validação e inferência, com histórico registrado.

### Segmentação

* Dois modos: SAM (automático, sem treino adicional) e métodos clássicos de visão computacional (Otsu, HSV, GrabCut, Watershed), com parâmetros ajustáveis;
* Modo comparativo entre os métodos, todos usando a bounding box do YOLO como ROI;
* Conversão de máscara para polígono, pronta para overlay no frontend.

### Aplicação Web

* API de inferência (detecção + segmentação) servindo um frontend;
* Upload de imagem ou seleção de imagens de teste por mob, visualização das detecções e das máscaras segmentadas.

## Tecnologias Utilizadas

* Python
* PyTorch
* Ultralytics YOLO e SAM
* FastAPI / Uvicorn
* PyQt6
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Jupyter Notebook
* Kaggle CLI
* Git e GitHub

## Roadmap

### Versão 1.0 — Detecção com YOLO

* [x] Estrutura inicial do projeto
* [x] Download automatizado do dataset
* [x] Análise exploratória dos dados
* [x] Verificação do balanceamento das classes
* [x] Validação visual das anotações
* [x] Treinamento baseline
* [x] Avaliação de desempenho
* [x] Testes de inferência
* [x] Comparação entre arquiteturas YOLO
* [x] Seleção do modelo final de detecção
* [ ] Validação cruzada (k-fold, k=5) — notebook pronto, execução pendente

### Versão 2.0 — Segmentação

* [x] Extração das regiões de interesse (ROI) via bounding boxes do YOLO
* [x] Integração YOLO + SAM (MobileSAM), sem treino adicional
* [x] Segmentação clássica (Otsu, HSV, GrabCut, Watershed) como modo alternativo/comparativo
* [x] Conversão de máscara para polígono
* [ ] Avaliação qualitativa comparando os métodos em mais classes

### Versão 3.0 — Aplicação Web

* [x] Desenvolvimento da API de inferência
* [x] Upload de imagens
* [x] Visualização das bounding boxes
* [x] Visualização das máscaras segmentadas
* [x] Imagens de teste servidas pela própria API, filtráveis por mob
* [ ] Aplicação web (em desenvolvimento em repositório separado)
* [ ] Dashboard de resultados
* [ ] Deploy em nuvem da API (Hugging Face Spaces)

## Resultados Esperados

* Alta precisão na identificação de mobs;
* Pipeline automatizado de treinamento e inferência;
* Base para futuros projetos envolvendo visão computacional em ambientes de jogos;
* Integração com aplicação web para inferência em tempo real.

## Licença

Este projeto é destinado a fins educacionais, pesquisa e aprendizado em visão computacional e Deep Learning.
