# 🏗️ Estrutura Organizada do Projeto YOLOCraft

## Estrutura Proposta

```
YOLOCraft/
│
├── 📋 README.md
├── 📋 CLAUDE.md
├── 📋 requirements.txt
├── 📋 .gitignore
│
├── 📁 data/
│   ├── minecraft_mobs/           # Dataset principal
│   │   └── minecraft_mobs_yolo/
│   │       ├── train/
│   │       ├── val/
│   │       └── data.yaml
│   └── test_samples/             # Imagens para teste
│       └── spiders/
│
├── 📁 notebooks/                 # Notebooks Jupyter organizados por fase
│   ├── 📁 1_exploracao/
│   │   ├── 01_dataset_analysis.ipynb
│   │   └── 02_visualize_labels.ipynb
│   │
│   ├── 📁 2_baseline/            # ✅ Seu notebook baseline aqui
│   │   ├── 01_baseline_training.ipynb
│   │   ├── 02_baseline_evaluation.ipynb
│   │   └── README_BASELINE.md    # Notas sobre o baseline
│   │
│   ├── 📁 3_experimentos/        # 🆕 Novos treinos e testes
│   │   ├── 01_model_size_comparison.ipynb
│   │   ├── 02_hyperparameter_tuning.ipynb
│   │   ├── 03_augmentation_tests.ipynb
│   │   └── README_EXPERIMENTOS.md
│   │
│   └── 📁 testes/               # Imagens de teste
│       └── spiders.jpg
│
├── 📁 src/                      # Código Python reutilizável
│   ├── __init__.py
│   ├── config.py               # Configurações do projeto
│   ├── training_logger.py      # Logger de treinos
│   ├── train_with_logging.py   # Script de treino
│   ├── test_thresholds.py      # Teste de thresholds
│   ├── train_improved.py       # Treinos melhorados
│   └── utils.py                # Funções auxiliares
│
├── 📁 models/                  # Modelos treinados organizados
│   ├── baseline/
│   │   └── best.pt
│   ├── experiments/
│   │   ├── exp_001_yolo26s/
│   │   │   └── best.pt
│   │   └── exp_002_augmentation/
│   │       └── best.pt
│   └── production/
│       └── best_model.pt
│
├── 📁 runs/                    # Outputs do YOLO (resultados de treino)
│   └── detect/
│       ├── train/
│       ├── train-3/
│       ├── val/
│       └── predict/
│
├── 📁 training_logs/           # Histórico de treinos (gerado automaticamente)
│   ├── training_history.json
│   └── training_history.csv
│
├── 📁 results/                 # Análises e resultados finais
│   ├── model_comparison.csv
│   ├── threshold_analysis.csv
│   └── visualizations/
│
├── 📁 threshold_tests/         # Testes de threshold (gerado por script)
│   └── conf_0.01.jpg
│
└── 📁 docs/                    # Documentação adicional
    ├── COMO_USAR_LOGGER.md
    ├── BEST_PRACTICES.md
    └── ROADMAP.md
```

---

## 📚 Organização de Notebooks

### 1️⃣ **Phase 1: Exploração** (`1_exploracao/`)
Análise inicial do dataset (NÃO MODIFICAR)
- Distribuição das classes
- Validação das anotações
- Estatísticas gerais

### 2️⃣ **Phase 2: Baseline** (`2_baseline/`)
Seu modelo baseline (USE COMO REFERÊNCIA)
- Treinamento com YOLO26n
- Validação
- Análise de resultados
- **README_BASELINE.md**: Documente suas decisões

### 3️⃣ **Phase 3: Experimentos** (`3_experimentos/`)
Novos treinos e melhorias (MODIFIQUE À VONTADE)
- Comparação de tamanhos de modelos
- Ajuste de hiperparâmetros
- Testes de augmentação
- Análise de thresholds

---

## 🏃 Como Usar

### Para um Novo Experimento:

**Passo 1:** Criar novo notebook em `3_experimentos/`

```
3_experimentos/
├── 01_model_size_comparison.ipynb  ← CRIAR AQUI
```

**Passo 2:** Estrutura do notebook

```python
"""
EXPERIMENTO: Comparação de Tamanhos de Modelos
DATA: 24/06/2024
OBJETIVO: Testar YOLO26s e YOLO26m
BASELINE: mAP50=0.9262, mAP50-95=0.7663
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd().resolve().parent.parent))

from src.training_logger import TrainingLogger
from src.train_improved import train_improved_model
import src.config as cfg

# Seu código aqui...

logger = TrainingLogger()
logger.print_summary()
```

**Passo 3:** Registre seus treinos

```python
logger.log_training(
    model_size="s",
    epochs=100,
    batch_size=16,
    imgsz=768,
    map50=0.9456,
    map50_95=0.7890,
    train_time=12000,
    notes="YOLO26s com augmentação padrão"
)
```

---

## 📊 Comparando Experimentos

Depois de vários treinos:

```python
from src.training_logger import TrainingLogger

logger = TrainingLogger()

# Ver todos
logger.print_summary()

# Comparar específico
logger.compare_models("n", "s")

# Melhor até agora
best = logger.export_best_model_info()
print(f"🏆 Melhor: YOLO26{best['model_size']} com mAP50={best['map50']}")
```

---

## 💾 Onde Salvar Modelos

### Estrutura de `models/`

```
models/
├── baseline/                    # Seu baseline (LOCK)
│   └── best.pt
│
├── experiments/                 # Seus testes
│   ├── exp_001_yolo26s/
│   │   ├── best.pt
│   │   ├── config.yaml
│   │   └── notes.txt
│   └── exp_002_augmentation/
│       └── best.pt
│
└── production/                  # Melhor modelo para usar
    └── best_model.pt           # Link simbólico para o melhor
```

### Como Salvar (no notebook):

```python
# Após treinar
best_pt = Path("runs/detect/train-4/weights/best.pt")
output = Path("models/experiments/exp_002_augmentation/")
output.mkdir(parents=True, exist_ok=True)

import shutil
shutil.copy(best_pt, output / "best.pt")

# Salvar notas
with open(output / "notes.txt", "w") as f:
    f.write("YOLO26s com 150 épocas\nmAP50=0.9456")
```

---

## 📝 Template de README para cada Experimento

**`3_experimentos/README_EXPERIMENTOS.md`**

```markdown
# 📊 Experimentos de Melhoria

## Exp 001: Comparação de Tamanhos (24/06/2024)

**Objetivo:** Comparar YOLO26n (baseline) com YOLO26s e YOLO26m

**Resultados:**
| Modelo | mAP50 | mAP50-95 | Tempo | Notas |
|--------|-------|----------|-------|-------|
| YOLO26n | 0.9262 | 0.7663 | 2.5h | Baseline |
| YOLO26s | 0.9456 | 0.7890 | 3.2h | ✅ Melhor! |
| YOLO26m | 0.9512 | 0.8010 | 4.5h | Muito lento |

**Conclusão:** YOLO26s é melhor que nano e rápido

---

## Exp 002: Augmentação Agressiva (25/06/2024)

...
```

---

## ✅ Checklist para Manter Organizado

- [ ] Cada novo experimento em notebook separado
- [ ] Nome descritivo: `NN_nome_do_experimento.ipynb`
- [ ] Documentar no início do notebook
- [ ] Usar `logger.log_training()` para registrar
- [ ] Salvar modelos interessantes em `models/experiments/`
- [ ] Atualizar README_EXPERIMENTOS.md com resultados

---

## 🎯 Benefícios dessa Estrutura

✅ **Rastreabilidade:** Cada experimento tem seu próprio notebook  
✅ **Comparação:** Fácil comparar baseline vs experimentos  
✅ **Reprodutibilidade:** Tudo documentado e versionado  
✅ **Escalabilidade:** Fácil adicionar novos experimentos  
✅ **Profissionalismo:** Estrutura clara e organizada  

---

## 🚀 Próximos Passos

1. Reorganize seus notebooks
2. Crie `3_experimentos/` e seu primeiro novo notebook
3. Use o `TrainingLogger` em todos os treinos
4. Mantenha `README_EXPERIMENTOS.md` atualizado
5. Depois de X experimentos, escolha o melhor e coloque em `models/production/`
