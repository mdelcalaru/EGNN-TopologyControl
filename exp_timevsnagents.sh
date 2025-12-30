#!/bin/bash
# Script para correr múltiples experimentos con distintos parámetros
# Guarda logs y corre análisis después de cada corrida

# Ruta base de checkpoints
CHECKPOINTS=./checkpoints

# Carpeta de resultados
RESULTS=$(readlink -f ./data_sync/resultados/multi_exp_timevsnagents)
#RESULTS=./data_sync/resultados

# Crear carpeta si no existe
mkdir -p "$RESULTS"

# Lista de pares TASK,COMM
PAIRS=("20 30")
#("3 2" "5 7" "7 12" "10 15" "12 18" "15 20" "15 25" "20 30")
# "8 12" "9 11" "7 18" "8 17" "9 16" "10 15")
#PAIRS=("10 15" "30 20" "40 60")
#"3 2" "10 12")

# Número fijo de experimentos
EXP=10
SEED=4321
for pair in "${PAIRS[@]}"; do
  # Leer TASK y COMM desde la tupla
  read TASK COMM <<< "$pair"

  echo "Ejecutando experimento con TASK=$TASK, COMM=$COMM, EXP=$EXP"
      
  LOGFILE="$RESULTS/run_task${TASK}_comm${COMM}_exp${EXP}.log"

  # Ejecutar experimento
  python -m performance_tests.multi_exp \
  --models real_5 $CHECKPOINTS/graph_model_NoGrad_playful-brook-55_model_all.ckpt $CHECKPOINTS/graph_model_Grad_laced-surf-105_model_all.ckpt \
           real_15 $CHECKPOINTS/graph_model_NoGrad_devoted-totem-58_model_all.ckpt $CHECKPOINTS/graph_model_Grad_easy-bush-108_model_all.ckpt \
  --task_agents $TASK \
  --comm_agents $COMM \
  --num_experiments $EXP \
  --experiment_steps 10 \
  --aproximations_steps 1 \
  --output $RESULTS \
  --distance_factor 0.8 \
  --dist delaunay \
  --step_size 0.2 \
  --seed $SEED \
  #\
  #> "$LOGFILE" 2>&1 
 
  # === Encontrar el último directorio creado (el de esta corrida) ===
  LAST_RUN=$(ls -td $RESULTS/*/ | head -1)

  echo ">>> Analizando resultados en $LAST_RUN"

  # Ejecutar análisis sobre el directorio
  python -m process_results.main_multi --data_dir "$LAST_RUN"
  curl -d "✅ Par $TASK task, $COMM net finalizado" https://ntfy.sh/gtc-runs-51288
  
done

curl -d "✅ Script multi_exp_31 finalizado" https://ntfy.sh/gtc-runs-51288
