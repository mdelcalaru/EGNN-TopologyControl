#!/bin/bash

# Número de repeticiones
NUM_RUNS=10
#20

# Número de muestras por ejecución
SAMPLES_PER_RUN=1250
#1250 

# Otros parámetros
EPSILON=0.1
LOCAL_SAMPLES=10
outDir="./data_sync/datasets/total_25/"
distance_factor=1.0
PAIRS=("7 18" "8 17" "9 16" "10 15" "11 14" "12 13" "13 12" "14 11" "15 10" "16 9")
#("3 1" "3 2" "3 3" "4 2" "4 3" "4 4" "4 5" "4 6" "5 2" "5 3" "5 4" "5 5" "5 6" "6 3" "6 4" "6 5" "6 6" "6 7" "6 8" "7 4" "7 5" "7 6" "7 8" "8 4" "8 5" "8 6" "8 7" "8 9")
for pair in "${PAIRS[@]}"; do
    read TASK COMM <<< "$pair"
    # Leer TASK y COMM desde la tupla
    echo "Generando muestras con con TASK=$TASK, COMM=$COMM"
    for i in $(seq 1 $NUM_RUNS); do
        echo "Run $i/$NUM_RUNS"
        if [ "$i" -le $((NUM_RUNS / 2)) ]; then

            python -m graph_dataset_generator.generate_dataset \
                --num_samples $SAMPLES_PER_RUN \
                --task_agents $TASK \
                --comm_agents $COMM \
                --local_samples $LOCAL_SAMPLES \
                --epsilon $EPSILON \
                --distance_factor $distance_factor \
                --out_dir $outDir\
                --dist 'poisson'
        else
            #Con delaunay
            python -m graph_dataset_generator.generate_dataset \
                --num_samples $SAMPLES_PER_RUN \
                --task_agents $TASK_AGENTS \
                --comm_agents $COMM_AGENTS \
                --local_samples $LOCAL_SAMPLES \
                --epsilon $EPSILON \
                --out_dir $outDir \
                --distance_factor $distance_factor \
                --dist 'delaunay'
        fi
        curl -d "✅ run $i Par $TASK task, $COMM net finalizado" https://ntfy.sh/gtc-runs-51288
    
    done
    curl -d "✅ Gen Par $TASK task, $COMM net finalizado" https://ntfy.sh/gtc-runs-51288

done
        
echo "All runs completed!"

curl -d "✅ generacion finalizado" https://ntfy.sh/gtc-runs-51288

 