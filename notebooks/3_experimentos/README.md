# 🧪 Experimentos - Melhorias no Modelo

Este diretório contém notebooks com experiências para melhorar o modelo baseline.

**Baseline (referência):** YOLO26n com mAP50=0.9262, mAP50-95=0.7663

---

## 📝 Como Começar um Novo Experimento

### Passo 1: Copie o Template
```bash
cp TEMPLATE_novo_experimento.ipynb 01_seu_experimento.ipynb
```

### Passo 2: Abra e Customize
1. Mude o título
2. Descreva o objetivo
3. Configure os parâmetros
4. Execute as células

### Passo 3: Registre os Resultados
O notebook automaticamente:
- ✅ Treina o modelo
- ✅ Registra as métricas
- ✅ Compara com baseline
- ✅ Salva o histórico em `training_logs/`

---

## 📊 Experimentos Realizados

| ID | Experimento | Modelo | mAP50 | mAP50-95 | Status |
|----|-----------|-|------|---|---|
| - | [Adicione aqui] | - | - | - | - |

---

## 🎯 Ideias para Experimentos

### 1. Comparação de Tamanhos ⭐
```python
# Testar YOLO26s e YOLO26m
MODEL_SIZE = "s"  # ou "m"
```

**Esperado:** Modelos maiores detectam melhor

### 2. Augmentação Agressiva
```python
# Aumentar data augmentation
hsv_h = 0.015
hsv_s = 0.7
hsv_v = 0.4
```

**Esperado:** Melhor generalização

### 3. Mais Épocas
```python
EPOCHS = 200  # em vez de 100
```

**Esperado:** Mais convergência = melhor mAP

### 4. Diferentes Learning Rates
```python
lr0 = 0.001  # ou 0.1, 0.05
```

**Esperado:** Encontrar taxa de aprendizado ideal

### 5. Transfer Learning (Fine-tuning)
```python
# Usar modelo pré-treinado
model = YOLO("yolo26s.pt")  # Maior que baseline
```

**Esperado:** Melhor performance, menos treino

---

## 🚀 Workflow Recomendado

1. **Comece simples:** YOLO26s vs YOLO26n
2. **Se melhorar:** Tente YOLO26m
3. **Otimize:** Ajuste hiperparâmetros
4. **Valide:** Teste com threshold_test.py
5. **Escolha o melhor:** Salve em `models/production/`

---

## 📁 Estrutura Local

```
3_experimentos/
├── README.md (você está aqui)
├── TEMPLATE_novo_experimento.ipynb (copie este)
├── 01_model_size_comparison.ipynb
├── 02_hyperparameter_tuning.ipynb
└── ...
```

---

## 💾 Arquivos Gerados

Cada notebook gera automaticamente:

```
../../training_logs/
├── training_history.json  # Seus registros
└── training_history.csv   # Visualização rápida

../../models/experiments/
├── exp_001_nome/
│   ├── best.pt
│   └── notes.txt
└── exp_002_nome/
    ├── best.pt
    └── notes.txt
```

---

## ⚡ Quick Start

**1. Copie o template:**
```bash
cp TEMPLATE_novo_experimento.ipynb 01_meu_teste.ipynb
```

**2. Abra no Jupyter e rode**

**3. Veja resultados:**
```python
from src.training_logger import TrainingLogger
logger = TrainingLogger()
logger.print_summary()
```

---

## 🎓 Aprendendo com Cada Experimento

Após cada experimento, pergunte-se:

✓ O modelo melhorou comparado ao baseline?  
✓ O tempo de treino vale a pena?  
✓ O modelo generalizou bem nos testes?  
✓ Qual foi o fator chave da melhoria?  

---

## 📞 Precisa de Ajuda?

- Veja o `TEMPLATE_novo_experimento.ipynb`
- Leia `../../COMO_USAR_LOGGER.md`
- Consulte `../../ESTRUTURA_PROJETO.md`

---

**Happy Experimenting! 🚀**
