#!/bin/bash
# Script para correr múltiples experimentos con distintos parámetros
# Guarda logs y corre análisis después de cada corrida

# Ruta base de checkpoints
CHECKPOINTS=./checkpoints

# Carpeta de resultados
RESULTS=./data_sync/resultados

# Crear carpeta si no existe
mkdir -p "$RESULTS"

# Lista de pares TASK,COMM
#PAIRS=("7 8")
PAIRS=("3 2" "7 5" "11 9" "17 13" "22 18" "28 22")

# Número fijo de experimentos
EXP=10
SEED=38
for pair in "${PAIRS[@]}"; do
  # Leer TASK y COMM desde la tupla
  read TASK COMM <<< "$pair"

  echo "Ejecutando experimento con TASK=$TASK, COMM=$COMM, EXP=$EXP"
      
  LOGFILE="$RESULTS/run_task${TASK}_comm${COMM}_exp${EXP}.log"

  # Ejecutar experimento
  python -m performance_tests.multi_exp \
  --models real_5 $CHECKPOINTS/graph_model_NoGrad_playful-brook-55_model_all.ckpt $CHECKPOINTS/graph_model_Grad_laced-surf-105_model_all.ckpt \
           real_25 $CHECKPOINTS/graph_model_NoGrad_deft-plant-70_model_all.ckpt $CHECKPOINTS/graph_model_Grad_clear-forest-117_model_all.ckpt \
  --task_agents $TASK \
  --comm_agents $COMM \
  --num_experiments $EXP \
  --experiment_steps 1 \
  --aproximations_steps 1\
  --output $RESULTS \
  --distance_factor 0.8\
  --dist poisson \
  --step_size 0.5 \
  --seed $SEED \
  #\
  #> "$LOGFILE" 2>&1 
 
  # === Encontrar el último directorio creado (el de esta corrida) ===
  LAST_RUN=$(ls -td $RESULTS/*/ | head -1)

  echo ">>> Analizando resultados en $LAST_RUN"

  # Ejecutar análisis sobre el directorio
  python -m process_results.main_multi --data_dir "$LAST_RUN"
  
done

