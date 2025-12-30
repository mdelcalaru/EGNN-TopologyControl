#!/bin/bash

# Número de repeticiones
NUM_RUNS=10

# Número de muestras por ejecución
SAMPLES_PER_RUN=1250 

# Otros parámetros
TASK_AGENTS=3
COMM_AGENTS=1
EPSILON=0.1
LOCAL_SAMPLES=10
outDir="./data_sync/datasets/"
distance_factor=0.8
# Ejecutar el script varias veces
for i in $(seq 1 $NUM_RUNS); do
    echo "Run $i/$NUM_RUNS"
    
    if [ "$i" -le $((NUM_RUNS / 2)) ]; then
       # Sin delaunay
       python -m graph_dataset_generator.generate_dataset \
           --num_samples $SAMPLES_PER_RUN \
           --task_agents $TASK_AGENTS \
           --comm_agents $COMM_AGENTS \
           --local_samples $LOCAL_SAMPLES \
           --epsilon $EPSILON \
           --distance_factor $distance_factor \
           --out_dir $outDir\
           --dist 'custom'
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
done
