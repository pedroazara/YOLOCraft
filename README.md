# YOLOCraft

Detecção e segmentação de mobs do Minecraft utilizando YOLO e visão computacional.

## Visão Geral

YOLOCraft é um projeto de Deep Learning voltado para a identificação automática de mobs do Minecraft em imagens. O objetivo é treinar modelos da família YOLO para detectar e/ou segmentar entidades do jogo, permitindo aplicações em visão computacional, aprendizado de máquina e experimentação com datasets de jogos.

O projeto foi desenvolvido como uma oportunidade de estudo prático em:

* Computer Vision
* Object Detection
* Instance Segmentation
* Treinamento de modelos YOLO
* Organização de pipelines de Machine Learning

## Dataset

O dataset utilizado é:

* Minecraft Mobs YOLO Dataset
* Fonte: Kaggle
* Classes correspondentes a diferentes mobs do Minecraft

O download pode ser realizado automaticamente através do script:

```bash
python download_dataset.py
```

## Estrutura do Projeto

```text
YOLOCraft/
│
├── data/
│   └── minecraft_mobs/
│
├── notebooks/
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── utils.py
│
├── models/
│
├── runs/
│
├── download_dataset.py
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

Linux / macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Download do Dataset

Após configurar suas credenciais do Kaggle:

```bash
python download_dataset.py
```

Os dados serão armazenados em:

```text
data/minecraft_mobs/
```

## Treinamento

Exemplo de treinamento com YOLO:

```bash
python src/train.py
```

Durante o treinamento serão gerados:

* Pesos do modelo
* Métricas de validação
* Curvas de aprendizado
* Resultados de inferência

## Inferência

Para realizar previsões em novas imagens:

```bash
python src/predict.py
```

Os resultados serão salvos no diretório:

```text
runs/
```

## Objetivos

* Detectar mobs automaticamente em imagens.
* Avaliar o desempenho de arquiteturas YOLO modernas.
* Explorar técnicas de segmentação e detecção de objetos.
* Desenvolver um pipeline reprodutível para projetos de visão computacional.

## Tecnologias Utilizadas

* Python
* PyTorch
* Ultralytics YOLO
* OpenCV
* NumPy
* Matplotlib
* Kaggle API

## Resultados Esperados

* Alta precisão na identificação de mobs.
* Pipeline automatizado de treinamento e inferência.
* Base para futuros projetos envolvendo visão computacional em ambientes de jogos.

## Licença

Este projeto é destinado a fins educacionais e de pesquisa.
