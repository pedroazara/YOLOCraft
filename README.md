# YOLOCraft

Detecção e segmentação de mobs do Minecraft utilizando YOLO, SAM e visão computacional.

## Visão Geral

YOLOCraft identifica automaticamente mobs do Minecraft em imagens. Um modelo YOLO detecta a classe e a bounding box de cada mob; o SAM (Segment Anything Model) usa essa box como prompt para gerar a máscara de segmentação do mob. Uma API expõe esse pipeline para uma aplicação web.

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

Um subconjunto de 16 classes (`data/minecraft_mobs-2/apresentacao`) foi curado com `src/dataset_manager.py` para o modelo de apresentação: cave_spider, creeper, enderman, skeleton, slime, spider, zombie, iron_golem, wolf, cat, chicken, cow, frog, horse, pig, sheep.

## Estado Atual do Modelo

* Modelo em treinamento (YOLO26s) no subconjunto curado de 16 classes — métricas parciais observadas: precision ≈ 0.98, recall ≈ 0.98, mAP50 ≈ 0.99, mAP50-95 ≈ 0.95.
* Modelo consolidado anterior (YOLO26s, 4 classes — creeper, skeleton, spider, zombie), 100 épocas: mAP50 = 0.9522, mAP50-95 = 0.8165.
* Histórico completo de treinos em `training_logs/training_history.csv`, gerado por `src/training_logger.py`.
* API (`src/api.py`) funcional: detecção (YOLO) + segmentação (MobileSAM) integradas no endpoint `/predict`.
* Frontend em desenvolvimento em repositório separado, consumindo a API pelo contrato descrito em `docs/frontend_integration.md`. Testado via túnel (ngrok) durante o desenvolvimento; deploy definitivo da API ainda pendente.

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
│   └── testes/                  # imagens de teste manual
│
├── src/
│   ├── config.py                # seleção de dataset (registro de paths)
│   ├── convert_dataset.py       # converte CSV de anotações para formato YOLO
│   ├── training_logger.py       # registra histórico de treinos (JSON/CSV)
│   ├── train_with_logging.py    # treino com registro automático
│   ├── train_improved.py        # treino com hiperparâmetros de augmentation
│   ├── test_thresholds.py       # varredura de confidence threshold
│   ├── detector_gui.py          # app desktop (PyQt6) para testar modelos
│   ├── dataset_manager.py       # app desktop (PyQt6) para curar o dataset
│   ├── api.py                   # API de inferência (FastAPI)
│   └── utils.py
│
├── scripts/
│   └── download_dataset.py      # download automatizado via Kaggle CLI
│
├── docs/
│   └── frontend_integration.md  # contrato da API para o frontend
│
├── pretrained_models/           # pesos pré-treinados (YOLO, MobileSAM)
├── training_logs/               # histórico de treinos
├── requirements.txt
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

Converta as anotações (CSV) para o formato YOLO com `src/convert_dataset.py`, ou use `src/dataset_manager.py` para selecionar classes e exportar um subconjunto curado.

## Treinamento

O treinamento é feito nos notebooks de `notebooks/3_experimentos/`, que registram cada execução via `src/training_logger.py`. Também há um ponto de entrada em script:

```bash
python -m src.train_with_logging
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

## Inferência

Duas formas de rodar inferência:

* **App desktop** (`src/detector_gui.py`): carrega um modelo `.pt`, permite ajustar o confidence threshold e testar imagens.
* **API** (`src/api.py`): endpoint `POST /predict`, recebe uma imagem e devolve as detecções (classe, confiança, box) e a segmentação (polígono da máscara, via SAM).

```bash
uvicorn src.api:app --reload --port 8000
python -m src.detector_gui
```

## Objetivos

### Detecção de Objetos

* Detectar mobs automaticamente em imagens utilizando modelos da família YOLO;
* Avaliar métricas como Precision, Recall, mAP50 e mAP50-95;
* Pipeline reproduzível de treinamento, validação e inferência, com histórico registrado.

### Segmentação

* Usar as bounding boxes do YOLO como prompt para o SAM (Segment Anything Model);
* Gerar máscaras de segmentação por instância, sem necessidade de treino adicional.

### Aplicação Web

* API de inferência (detecção + segmentação) servindo um frontend;
* Upload de imagem, visualização das detecções e das máscaras segmentadas.

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

### Versão 2.0 — Segmentação

* [x] Extração das regiões de interesse (ROI) via bounding boxes do YOLO
* [x] Integração YOLO + SAM (MobileSAM), sem treino adicional
* [x] Conversão de máscara para polígono
* [ ] Avaliação qualitativa dos resultados em mais classes

Abordagem original (segmentação clássica com OpenCV — Threshold, Otsu, Canny, GrabCut, Watershed) foi substituída pelo SAM, que generaliza sem necessidade de ajuste manual por classe.

### Versão 3.0 — Aplicação Web

* [x] Desenvolvimento da API de inferência
* [x] Upload de imagens
* [x] Visualização das bounding boxes
* [x] Visualização das máscaras segmentadas
* [ ] Aplicação web (em desenvolvimento em repositório separado)
* [ ] Dashboard de resultados
* [ ] Deploy em nuvem da API

## Resultados Esperados

* Alta precisão na identificação de mobs;
* Pipeline automatizado de treinamento e inferência;
* Base para futuros projetos envolvendo visão computacional em ambientes de jogos;
* Integração com aplicação web para inferência em tempo real.

## Licença

Este projeto é destinado a fins educacionais, pesquisa e aprendizado em visão computacional e Deep Learning.
