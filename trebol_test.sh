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
PAIRS=("3 2")
#"3 2" "10 12")


for pair in "${PAIRS[@]}"; do
  # Leer TASK y COMM desde la tupla
  read TASK COMM <<< "$pair"

  echo "Ejecutando experimento con TASK=$TASK, COMM=$COMM"
      
  LOGFILE="$RESULTS/run_task${TASK}_comm${COMM}_trebol.log"

  # Ejecutar experimento
  python -m performance_tests.trebol_experiment \
  --models real_5 $CHECKPOINTS/graph_model_NoGrad_playful-brook-55_model_all.ckpt $CHECKPOINTS/graph_model_Grad_laced-surf-105_model_all.ckpt \
  --task_agents $TASK \
  --comm_agents $COMM \
  --aproximations_steps 5 \
  --output $RESULTS \
  --distance_factor 0.55\
  #\
  #> "$LOGFILE" 2>&1 

  #echo "    → Log guardado en $LOGFILE"

  # === Encontrar el último directorio creado (el de esta corrida) ===
  LAST_RUN=$(ls -td $RESULTS/*/ | head -1)

  echo ">>> Analizando resultados en $LAST_RUN"

  # Ejecutar análisis sobre el directorio
  python -m process_results.main_multi --data_dir "$LAST_RUN"
  curl -d "✅ Par $TASK task, $COMM net finalizado" https://ntfy.sh/gtc-runs-51288
  
done

curl -d "✅ Script trebol test finalizado" https://ntfy.sh/gtc-runs-51288
