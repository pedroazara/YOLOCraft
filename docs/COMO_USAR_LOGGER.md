# 📊 Como Usar o Training Logger

O sistema de logging automático rastreia todos os seus treinos e os organiza em um arquivo JSON e CSV.

## 🚀 Uso Básico

### Opção 1: No seu Notebook

```python
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd().resolve().parent))

from src.training_logger import TrainingLogger
from ultralytics import YOLO
import torch
import time

# Inicializar logger
logger = TrainingLogger(log_dir="training_logs")

# Treinar modelo
model = YOLO("yolo26n.pt")

start_time = time.time()

results = model.train(
    data="data/minecraft_mobs/minecraft_mobs_yolo/data.yaml",
    epochs=100,
    imgsz=768,
    batch=16,
)

train_time = time.time() - start_time

# Validar
metrics = model.val(workers=0)

# ✅ REGISTRAR O TREINAMENTO
train_id, record = logger.log_training(
    model_size="n",
    epochs=100,
    batch_size=16,
    imgsz=768,
    map50=metrics.box.map50,
    map50_95=metrics.box.map,
    train_time=train_time,
    notes="Primeiro treino baseline com YOLO26n"
)

# Mostrar histórico
logger.print_summary()
```

### Opção 2: Usar o Script Pronto

```bash
python src/train_with_logging.py
```

---

## 📋 Funções Disponíveis

### 1. Ver Resumo de Todos os Treinos
```python
logger = TrainingLogger()
logger.print_summary()
```

**Saída:**
```
================================================================================
📊 HISTÓRICO DE TREINOS
================================================================================
ID                 Modelo     Épocas   mAP50      mAP50-95   Tempo (h)  Data/Hora
--------------------------------------------------------------------------------
20240624_235930    YOLO26n    100      0.9262     0.7663     2.50       24/06/2024
20240623_180400    YOLO26s    100      0.9456     0.7890     3.20       23/06/2024

🏆 MELHOR MODELO: YOLO26s (ID: 20240623_180400)
   mAP50:    0.9456
   mAP50-95: 0.7890
================================================================================
```

### 2. Ver Detalhes de um Treino Específico
```python
logger.print_detailed("20240624_235930")
```

**Saída:**
```
============================================================
📋 DETALHES DO TREINO: 20240624_235930
============================================================

📅 Data/Hora: 24/06/2024 23:59:30
🤖 Modelo: YOLO26n

⚙️  Configuração:
   • Épocas: 100
   • Batch Size: 16
   • Tamanho da Imagem: 768
   • GPU: NVIDIA GeForce RTX 4060 Ti

📊 Resultados:
   • mAP50: 0.9262
   • mAP50-95: 0.7663

⏱️  Tempo de Treino:
   • 9000.00 segundos
   • 2.50 horas

📝 Notas: Primeiro treino baseline com YOLO26n
============================================================
```

### 3. Obter o Melhor Modelo
```python
best_model = logger.get_best_model("map50")
print(f"Melhor modelo: YOLO26{best_model['model_size']}")
print(f"mAP50: {best_model['map50']}")
```

### 4. Ver Todos os Treinos de um Tamanho Específico
```python
nano_models = logger.get_models_by_size("n")
print(f"Total de treinos YOLO26n: {len(nano_models)}")
```

### 5. Comparar Diferentes Tamanhos
```python
logger.compare_models("n", "s")
```

### 6. Exportar Info do Melhor Modelo
```python
best_info = logger.export_best_model_info()
print(best_info)
# {
#   'model_id': '20240623_180400',
#   'model_size': 's',
#   'map50': 0.9456,
#   'map50_95': 0.7890,
#   ...
# }
```

---

## 📁 Arquivos Gerados

Os logs são salvos em `training_logs/`:

```
training_logs/
├── training_history.json    # Dados completos em JSON
└── training_history.csv     # Visualização em CSV
```

### training_history.json
```json
[
  {
    "id": "20240624_235930",
    "timestamp": "2024-06-24T23:59:30.123456",
    "date": "24/06/2024",
    "time": "23:59:30",
    "model_size": "n",
    "epochs": 100,
    "batch_size": 16,
    "imgsz": 768,
    "map50": 0.9262,
    "map50_95": 0.7663,
    "train_time_seconds": 9000,
    "train_time_hours": 2.5,
    "gpu": "NVIDIA GeForce RTX 4060 Ti",
    "notes": "Primeiro treino baseline",
    "hyperparameters": {
      "learning_rate": "0.01",
      "optimizer": "SGD"
    }
  }
]
```

---

## 💡 Dicas Úteis

### 1. Comparar Performance de Modelos
```python
logger = TrainingLogger()

# Todos os YOLO26n
nano = logger.get_models_by_size("n")
print(f"Melhor YOLO26n: {max(nano, key=lambda x: x['map50'])['map50']:.4f}")

# Todos os YOLO26s
small = logger.get_models_by_size("s")
print(f"Melhor YOLO26s: {max(small, key=lambda x: x['map50'])['map50']:.4f}")
```

### 2. Verificar Qual Foi Seu Melhor Treino
```python
logger = TrainingLogger()
best = logger.get_best_model("map50")
logger.print_detailed(best['id'])
```

### 3. Acompanhar Melhorias ao Longo do Tempo
```python
logger = TrainingLogger()

for record in logger.history:
    print(f"{record['date']} - mAP50: {record['map50']:.4f}")
```

---

## 🎯 Próximos Passos

Depois de registrar vários treinos, você pode:

1. **Decidir qual modelo usar** baseado nos logs
2. **Comparar diferentes hiperparâmetros** 
3. **Acompanhar progresso ao longo do tempo**
4. **Reproduzir treinos bem-sucedidos** usando os registros

---

## ❓ Dúvidas?

Se precisar de ajuda, consulte:
- `src/training_logger.py` - Código completo da classe
- `src/train_with_logging.py` - Exemplo de integração
- `training_logs/training_history.json` - Seus dados brutos
