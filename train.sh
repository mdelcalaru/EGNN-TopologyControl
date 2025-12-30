#!/bin/bash
# Script para correr múltiples experimentos con distintos parámetros
# Guarda logs y corre análisis después de cada corrida

# Ruta base de checkpoints
DATA_DIRECTORY="./data_sync/datasets/total_25/"

# Entrenar modelo sin gradientes
python -m train --use_all --data_dir $DATA_DIRECTORY 

curl -d "✅ Entrenamiento NoGrad Finalizado" https://ntfy.sh/gtc-runs-51288

# Entrenar modelo con gradientes
python -m train --use_all --data_dir $DATA_DIRECTORY --grads 

curl -d "✅ Entrenamiento Grad Finalizado" https://ntfy.sh/gtc-runs-51288


