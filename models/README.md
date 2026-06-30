# 🤖 Modelos

Estrutura de modelos treinados do projeto YOLOCraft.

## Pastas

### baseline/
Seu modelo baseline (CONGELADO)
- Modelo: YOLO26n
- mAP50: 0.9262
- mAP50-95: 0.7663

### experiments/
Modelos dos experimentos
- exp_001_yolo26s/ (se existir)
- exp_002_.../ (adicione novos)

### production/
Melhor modelo para usar em produção
- best_model.pt (atualizar quando achar um melhor)

---

**Como adicionar um novo modelo:**

1. Treine no notebook em 
otebooks/3_experimentos/
2. Copie o est.pt para experiments/exp_XXX_nome/
3. Adicione um 
otes.txt com detalhes
