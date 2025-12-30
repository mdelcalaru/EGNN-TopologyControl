import argparse
import os
import pickle as pk
from datetime import datetime
from collections import Counter
from pathlib import Path
from process_results.utils_io import load_log_files, load_log_file
from process_results.analysis import compute_relative_differences_static
from process_results.plotting import plot_comparison


def main(data_dir=None, input_file=None, output_base=None):
    # Validación de entrada
    if not data_dir and not input_file:
        print("[ERROR] Debes especificar un directorio con --data_dir o un archivo con --input_file")
        return
    if data_dir and not os.path.isdir(data_dir):
        print(f"[ERROR] {data_dir} no es un directorio válido.")
        return
    if input_file and not os.path.isfile(input_file):
        print(f"[ERROR] {input_file} no es un archivo válido.")
        return

    # Preparar carpeta de salida
    # Determinar el directorio base de salida
    if data_dir:
        base_output_dir = Path(data_dir)
    else:
        base_output_dir = Path(input_file).parent

    # Crear subcarpeta timestamp dentro del mismo directorio
    output_path = base_output_dir / f"processed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path.mkdir(exist_ok=True)
    
    if output_base:
        print(f"[INFO] Se usará '{output_base}' como base para guardar datos.")
    else:
        output_base = "resultados"

    # Cargar datos
    if data_dir:
        log_data= load_log_files(data_dir)
    else:
        log_data, meta  = load_log_file(input_file)

    if not log_data:
        print("[ERROR] No se cargaron datos.")
        return

    # Resumen de keys
    print(f"[INFO] Número de experimentos cargados: {len(log_data)}")
    all_keys = []
    for exp_id, data in enumerate(log_data):
        keys = list(data.keys())
        all_keys.extend(keys)
        print(f"  - Experimento {exp_id}: keys = {keys}")
    
    key_counts = Counter(all_keys)
    print("\n[INFO] Resumen de todas las keys encontradas:")
    for key, count in key_counts.items():
        print(f"  - '{key}': {count} veces")
    if meta is not None:
        print(f"[INFO] Metadata found:")
        for k, v in meta.items():
            print(f"  - {k}: {v}")
    # Guardar resumen en info.txt
    info_path = os.path.join(output_path, "info.txt")
    with open(info_path, "w") as f:
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if data_dir:
            f.write(f"Fuente: Directorio - {data_dir}\n")
        elif input_file:
            f.write(f"Fuente: Archivo - {input_file}\n")
        
        f.write(f"Número de experimentos cargados: {len(log_data)}\n")
        for exp_id, data in enumerate(log_data):
            keys = list(log_data[0].keys())
            f.write(f"  - Experimento {exp_id}: keys = {keys}\n")
        
        f.write("\nResumen de todas las keys encontradas:\n")
        for key, count in key_counts.items():
            f.write(f"  - '{key}': {count} veces\n")
        if meta is not None:
            f.write("\nMetadata found:\n")
            for k, v in meta.items():
                f.write(f"  - {k}: {v}\n")
                
    # Procesamiento y visualización
    results = compute_relative_differences_static(log_data)
    # Guardar datos procesados
    with open(os.path.join(output_path, "processed.pkl"), "wb") as f:
        pk.dump(results, f)
        print(f"[INFO] Resultados procesados guardados en: {os.path.join(output_path, 'processed.pkl')}")

    plot_comparison(results, os.path.join(output_path, output_base))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesar y visualizar resultados de configuraciones")
    parser.add_argument("--data_dir", type=str, help="Directorio donde están los archivos .pkl")
    parser.add_argument("--input_file", type=str, help="Archivo .pkl con todos los resultados")
    parser.add_argument("--output_base", type=str, default=None, help="Nombre base para guardar los datos")

    args = parser.parse_args()
    main(data_dir=args.data_dir, input_file=args.input_file, output_base=args.output_base)
