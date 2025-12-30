import argparse
import os
import pickle as pk
from datetime import datetime
from collections import Counter
from pathlib import Path
from process_results.utils_io import load_log_files, load_log_file
from process_results.analysis import (
    compute_relative_differences_dynamic,
    summarize_dynamic_results,
    compute_relative_differences_static
)
from process_results.plotting import plot_dynamic_summary, plot_comparison_variants, animar_trayectorias_con_metricas, plot_y_evolution


def main(data_dir=None, output_base=None, excluded_variants=None):
    # Validación de entrada
    if not data_dir or not os.path.isdir(data_dir):
        print(f"[ERROR] '{data_dir}' no es un directorio válido.")
        return

    base_folder = Path(data_dir)
    # Crear subcarpeta timestamp dentro del mismo directorio
    output_path = base_folder / f"processed_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    output_path.mkdir(exist_ok=True)

    if output_base:
        print(f"[INFO] Se usará '{output_base}' como base para guardar datos.")
    else:
        output_base = "resultados"

    # Separar archivos por tipo según el nombre
    static_files = []
    dynamic_files = []
    trebol_files = []
    for file in base_folder.glob("*.pkl"):
        fname = file.name.lower()
        if "static" in fname:
            static_files.append(file)
        elif "dynamic" in fname:
            dynamic_files.append(file)
        elif "trebol" in fname:
            trebol_files.append(file)  # Asumir estático si no se especifica

    # Guardar info.txt
    info_path = output_path / "info.txt"
    with open(info_path, "w") as f:
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Fuente: Directorio - {data_dir}\n")
        f.write(f"Número de archivos cargados: {len(static_files) + len(dynamic_files)}\n")
        f.write(f"  - Archivos estáticos: {len(static_files)}\n")
        f.write(f"  - Archivos dinámicos: {len(dynamic_files)}\n")
        f.write("\nLista de archivos:\n")
        for file in static_files + dynamic_files:
            f.write(f"  - {file.name}\n")

    # Procesar archivos estáticos
    if static_files:
        print(f"[INFO] Procesando {len(static_files)} archivos estáticos...")
        log_data_static = load_log_files(static_files)
        results_static = compute_relative_differences_static(log_data_static, excluded_variants=excluded_variants)
        plot_comparison_variants(results_static, output_path)
        # Guardar datos procesados
        with open(output_path / "processed_static.pkl", "wb") as f:
            pk.dump(results_static, f)
            print(f"[INFO] Resultados estáticos guardados en: {output_path / 'processed_static.pkl'}")

    # Procesar archivos dinámicos
    if dynamic_files:
        print(f"[INFO] Procesando {len(dynamic_files)} archivos dinámicos...")
        log_data_dynamic = load_log_files(dynamic_files)
        results_dynamic = compute_relative_differences_dynamic(log_data_dynamic)
        summary_dynamic = summarize_dynamic_results(results_dynamic)
        animar_trayectorias_con_metricas(log_data_dynamic, filename=output_path / "trajectories_animation.mp4", fps=2, excluded_variants=excluded_variants)
        plot_y_evolution(log_data_dynamic, outputPath=output_path)
        plot_dynamic_summary(summary_dynamic, output_path)
        # Guardar datos procesados
        with open(output_path / "processed_dynamic.pkl", "wb") as f:
            pk.dump(results_dynamic, f)
            print(f"[INFO] Resultados dinámicos guardados en: {output_path / 'processed_dynamic.pkl'}")
    if trebol_files:
        print(f"[INFO] Procesando {len(trebol_files)} archivos trebol...")
        log_data_dynamic = load_log_files(trebol_files)
        results_dynamic = compute_relative_differences_dynamic(log_data_dynamic)
        summary_dynamic = summarize_dynamic_results(results_dynamic)
        animar_trayectorias_con_metricas(log_data_dynamic, filename=output_path / "trajectories_animation_trebol.mp4", fps=2, excluded_variants=excluded_variants)
        plot_dynamic_summary(summary_dynamic, output_path)
        # Guardar datos procesados
        with open(output_path / "processed_trebol.pkl", "wb") as f:
            pk.dump(results_dynamic, f)
            print(f"[INFO] Resultados dinámicos guardados en: {output_path / 'processed_trebol.pkl'}")
    print(f"[INFO] Procesamiento terminado. Resultados guardados en {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesar y visualizar resultados de configuraciones")
    parser.add_argument("--data_dir", type=str, required=True, help="Directorio donde están los archivos .pkl")
    parser.add_argument("--output_base", type=str, default=None, help="Nombre base para guardar los datos")
    parser.add_argument("--exclude", type=str, nargs='*', default=None, help="Lista de variantes a excluir del análisis")
    args = parser.parse_args()
    main(data_dir=args.data_dir, output_base=args.output_base, excluded_variants=args.exclude)
