# YOLOCraft

Detecção e segmentação de mobs do Minecraft utilizando YOLO e visão computacional.

## Visão Geral

YOLOCraft é um projeto de Deep Learning voltado para a identificação automática de mobs do Minecraft em imagens. O objetivo é treinar modelos da família YOLO para detectar e/ou segmentar entidades do jogo, permitindo aplicações em visão computacional, aprendizado de máquina e experimentação com datasets de jogos.

Atualmente, o foco do projeto está no treinamento, avaliação e testes de modelos de detecção e segmentação. Em versões futuras, o modelo treinado será integrado a uma aplicação web para inferência em tempo real.

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

Dataset:

https://www.kaggle.com/datasets/dracotlw/minecraft-mobs-yolo-dataset/data

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
python download_dataset.py
```

O script irá:

* Verificar se o dataset já existe localmente;
* Fazer o download apenas quando necessário;
* Organizar automaticamente os arquivos do projeto.

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

Os resultados serão salvos em:

```text
runs/
```

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

### Fase Atual

* Detectar mobs automaticamente em imagens;
* Avaliar o desempenho de arquiteturas YOLO modernas;
* Explorar técnicas de segmentação e detecção de objetos;
* Desenvolver um pipeline reproduzível para projetos de visão computacional;
* Comparar diferentes modelos YOLO e hiperparâmetros.

### Fase Futura

* Desenvolver uma aplicação web para inferência;
* Permitir upload de imagens pelo usuário;
* Exibir detecções em tempo real;
* Disponibilizar o modelo treinado online;
* Criar uma interface amigável para demonstração dos resultados.

## Tecnologias Utilizadas

* Python
* PyTorch
* Ultralytics YOLO
* OpenCV
* NumPy
* Matplotlib
* Kaggle CLI

## Roadmap

### Versão 1.0

* [x] Estrutura inicial do projeto
* [x] Download automatizado do dataset
* [ ] Análise exploratória dos dados
* [ ] Treinamento do modelo
* [ ] Avaliação de desempenho
* [ ] Testes de inferência

### Versão 2.0

* [ ] Otimização de hiperparâmetros
* [ ] Data augmentation
* [ ] Comparação entre arquiteturas YOLO
* [ ] Exportação do modelo treinado

### Versão 3.0

* [ ] API para inferência
* [ ] Aplicação web
* [ ] Upload de imagens
* [ ] Interface para visualização das detecções
* [ ] Deploy em nuvem

## Resultados Esperados

* Alta precisão na identificação de mobs;
* Pipeline automatizado de treinamento e inferência;
* Base para futuros projetos envolvendo visão computacional em ambientes de jogos;
* Integração futura com aplicações web.

## Licença

Este projeto é destinado a fins educacionais, pesquisa e aprendizado em visão computacional e Deep Learning.
