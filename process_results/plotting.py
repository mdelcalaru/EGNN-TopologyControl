import matplotlib.pyplot as plt
from itertools import cycle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from matplotlib import gridspec

# Colores fijos para ini y maxL2
fixed_colors = {
    'maxL2': 'lightgreen'
}

def plot_comparison_variants(results, output_path):
    
    variants = set()
    ref_variants = set()
    for k in results:
        if k.startswith('mnf-ini-'):
            variant = k.replace('mnf-ini-', '')
            variants.add(variant)
        elif k.startswith('mnf-'):
            # Extraer el nombre de la referencia
            # Forma: mnf_<ref>_<variant>
            parts = k.split('-')
            if len(parts) >= 3:
                ref_variants.add(parts[1])
                variants.add(parts[2])

    
    print("Referencias detectadas:", ref_variants)
    print("Variantes detectadas:", variants)
    # Grupos a plotear
    plots_num = 2 * len(ref_variants) + 1
    grupos = {
        'INI': {'prefix': 'mnf-ini-', 'ylabel': '% vs ini', 'titulo': 'Mejora sobre configuración inicial'}}
    for ref in ref_variants:
        grupos[f'ref']= {'prefix': f'mnf-{ref}-', 'ylabel': f'% vs {ref}', 'titulo': f'Diferencia respecto a $\lambda_2$ {ref}'}
        grupos[f'time_']= {'prefix': f'time-', 'ylabel': f'% vs {ref}', 'titulo': f'Dif en Tiempo de cómputo relativo a {ref}'}
    
    
    fig, ax = plt.subplots(plots_num, 1, figsize=(10, 12))
  # Colores consistentes para todas las variantes
    auto_colors = cycle([c for c in cm.get_cmap('tab10').colors])
    variant_colors = {}
    for label in sorted(variants):
        if label in fixed_colors:
            variant_colors[label] = fixed_colors[label]
        else:
            variant_colors[label] = next(auto_colors)

    # Plot
    for i, (grupo, props) in enumerate(grupos.items()):
        prefix = props['prefix']
        keys = [k for k in results if k.startswith(prefix)]
        if not keys:
            print(f"No data for {grupo}")
            continue

        data = [results[k] for k in keys]
        labels = [k.replace(prefix, '') for k in keys]
        labelsplt = [label.replace('_modelY','Y').replace('_modelG','G') for label in labels]
        colores = [variant_colors[label] for label in labels]

        bplot = ax[i].boxplot(data, patch_artist=True, showfliers=False,
                              medianprops={'color': 'darkblue'})
        ax[i].axhline(0, color='gray', linestyle='--', linewidth=1)
        ax[i].set_facecolor("white")
        ax[i].set_ylabel(props['ylabel'])
        ax[i].set_title(props['titulo'])
        ax[i].set_xticks(range(1, len(labels) + 1))
        ax[i].set_xticklabels(labelsplt, fontsize=11)
        ax[i].grid(axis='y', linestyle='--', alpha=0.7)

        for patch, color in zip(bplot['boxes'], colores):
            patch.set_facecolor(color)

    plt.tight_layout()
    plt.savefig(output_path / 'boxplots.pdf', dpi=300)
    plt.savefig(output_path / 'boxplots.png', dpi=300)
    plt.close(fig)
    
    print(f"Plots saved to {output_path}/boxplots.pdf/png")

import matplotlib.pyplot as plt

def plot_dynamic_summary(summary, output_path):
    '''summary is a dictionary:
    variant: (mean, std)
    '''
    title= r"Relative error to $\lambda_2$ Max"
    ylabel="Relative error w.r.t. Max $\lambda_2$ [%]"
    xlabel="Step"
    #ylim=(0, 1.05)
    
    
    for key, data in summary.items():
        steps = range(len(data[0]))
        plt.figure(figsize=(8, 5))
        plt.plot(steps, data[0], label=key)
        plt.fill_between(steps,
                         data[0] - data[1],
                         data[0] + data[1],
                         alpha=0.2)
    
        plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)  # línea horizontal en 0
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        #plt.ylim(*ylim)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / f'{key}_dynamic.pdf', dpi=300)
        plt.savefig(output_path / f'{key}_dynamic.png', dpi=300)
        plt.close()



def obtener_limites(hist_list):
    all_points = np.vstack([np.vstack(frames) for frames in hist_list])
    xmin, ymin = np.min(all_points, axis=0)
    xmax, ymax = np.max(all_points, axis=0)
    return xmin, xmax, ymin, ymax

def animar_trayectorias_con_metricas(experiment_data, filename=None, fps=5, excluded_variants=None):
    
    # --- misma estructura de preparación de resultados ---
    if 'modelos' in experiment_data[0].keys():      
        resultados_lista = [{**{k: d[k] for k in ["free_hist", "maxL2", "noSA"] if k in d},
                            **{f"{modelo}_{subkey}": valor
                                for modelo, subdict in d.get("modelos", {}).items()
                                for subkey, valor in subdict.items()
                            }
                            }
                            for d in experiment_data]
    else:
        resultados_lista = experiment_data

    variants = [key for key in resultados_lista[0].keys() if key not in {'ini', 'free_hist'}.union(excluded_variants or [])]
    print("animation Variants found:", variants)
    
    if 'maxL2' not in experiment_data[0] :
        print(f"[WARN] Al menos 1 Experimento: faltan datos de maxL2.")
    else:
        if experiment_data[0]['maxL2'] is None:
            variants.pop('maxL2', None)
            print(f"[WARN] Al menos 1 Experimento: faltan datos de maxL2.")

        
    titles = [var.replace('_modelY','Y').replace('_modelG','G') for var in variants]    
    for exp, data in enumerate(resultados_lista):
        if 'free_hist' not in data or data['free_hist'] is None:
            print(f"[WARN] Experimento {exp}: faltan datos de free_hist.")
            continue
        
        n_iter = len(data['free_hist'])
        n_metricas = len(variants)

        fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(2, n_metricas, height_ratios=[3, 1])

        traj_axes = [fig.add_subplot(gs[0, j]) for j in range(n_metricas)]
        metric_ax = fig.add_subplot(gs[1, :])  # ocupa todas las columnas

        x_mins, x_maxs, y_mins, y_maxs = [], [], [], []
        
        xmin_var, xmax_var, ymin_var, ymax_var = obtener_limites([data['free_hist']])
        x_mins.append(xmin_var); x_maxs.append(xmax_var)
        y_mins.append(ymin_var); y_maxs.append(ymax_var)

        mins_metricas, maxs_metricas = [], []
        for var in variants:
            xmin_var, xmax_var, ymin_var, ymax_var = obtener_limites([data[var]['comm_config_hist']])
            x_mins.append(xmin_var); x_maxs.append(xmax_var)
            y_mins.append(ymin_var); y_maxs.append(ymax_var)

            mins_metricas.append(np.min(data[var]['MNF_hist']))
            maxs_metricas.append(np.max(data[var]['MNF_hist']))

        xmin, xmax = min(x_mins), max(x_maxs)
        ymin, ymax = min(y_mins), max(y_maxs)
        margen_x, margen_y = (xmax - xmin) * 0.1, (ymax - ymin) * 0.1

        ymin_metricas, ymax_metricas = min(mins_metricas), max(maxs_metricas)
        margen_metrica = (ymax_metricas - ymin_metricas) * 0.1
       
        sc_free, sc_comm = [], []

        for ax, title in zip(traj_axes, titles):
            ax.set_title(title)
            ax.set_xlim(xmin - margen_x, xmax + margen_x)
            ax.set_ylim(ymin - margen_y, ymax + margen_y)
            sc_free.append(ax.scatter([], [], c='blue', label='Tarea'))
            sc_comm.append(ax.scatter([], [], c='red', label='Comunic.'))
            ax.legend()

        metric_ax.set_title("Evolución de la métrica")
        metric_ax.set_xlim(0, n_iter)
        metric_ax.set_ylim(ymin=ymin_metricas - margen_metrica, ymax=ymax_metricas + margen_metrica)
        metric_ax.set_xlabel("Iteración")
        metric_ax.set_ylabel("Valor")
        lines = []
        for var in variants:
            metricas = data[var]['MNF_hist']
            line, = metric_ax.plot([], [], label=var.replace('_modelY','Y').replace('_modelG','G'))
            lines.append((line, metricas))
        
        metric_ax.legend()

        def init():
            empty = np.empty((0, 2))
            for s1, s2 in zip(sc_free, sc_comm):
                s1.set_offsets(empty)
                s2.set_offsets(empty)
            for line, _ in lines:
                line.set_data([], [])
            return sc_free + sc_comm + [line for line, _ in lines]

        def update(frame):
            for j, var in enumerate(variants):
                free_pos = data['free_hist'][frame]
                comm_pos = data[var]['comm_config_hist'][frame]
                sc_free[j].set_offsets(free_pos)
                sc_comm[j].set_offsets(comm_pos)

            x = np.arange(frame + 1)
            for (line, metrica) in lines:
                line.set_data(x, metrica[:frame + 1])
            
            return sc_free + sc_comm + [line for line, _ in lines]

        ani = animation.FuncAnimation(fig, update, frames=n_iter, init_func=init,
                                    interval=300, blit=True)

        plt.tight_layout()

        # --- guardar un archivo distinto por experimento ---
        if filename:
            base, ext = os.path.splitext(filename)
            exp_filename = f"{base}_exp{exp}{ext}"
            if ext == ".gif":
                ani.save(exp_filename, writer='pillow', fps=fps)
            elif ext == ".mp4":
                ani.save(exp_filename, writer='ffmpeg', fps=fps)
            else:
                raise ValueError("El archivo debe ser .mp4 o .gif")
            print(f"Animación guardada como '{exp_filename}'")
        else:
            print("No se guardó la animación porque filename es None")

        plt.close(fig)

def animar_trayectorias_con_metricas0(experiment_data, filename=None, fps=5, excluded_variants=None):
    
    # --- misma estructura de preparación de resultados ---
    if 'modelos' in experiment_data[0].keys():      
        resultados_lista = [{**{k: d[k] for k in ["free_hist", "maxL2", "noSA"] if k in d},
                            **{f"{modelo}_{subkey}": valor
                                for modelo, subdict in d.get("modelos", {}).items()
                                for subkey, valor in subdict.items()
                            }
                            }
                            for d in experiment_data]
    else:
        resultados_lista = experiment_data

    variants = [key for key in resultados_lista[0].keys() if key not in {'ini', 'free_hist'}.union(excluded_variants or [])]
    titles = [var.replace('_modelY','Y').replace('_modelG','G') for var in variants]

    for exp, data in enumerate(resultados_lista):
        if 'free_hist' not in data or data['free_hist'] is None:
            continue

        n_iter = len(data['free_hist'])
        n_metricas = len(variants)

        fig = plt.figure(figsize=(15, 8))
        gs = gridspec.GridSpec(2, n_metricas, height_ratios=[3, 1])
        traj_axes = [fig.add_subplot(gs[0, j]) for j in range(n_metricas)]
        metric_ax = fig.add_subplot(gs[1, :])

        # --- Scatter y líneas de colas por panel ---
        sc_free, sc_comm = [], []
        lines_free, lines_comm = [], []
        for ax in traj_axes:
            ax.set_title("Trayectorias")
            ax.set_xlim(0, 10)  # ajustar según necesidad
            ax.set_ylim(0, 10)
            sc_free.append(ax.scatter([], [], c='blue', label='Tarea'))
            sc_comm.append(ax.scatter([], [], c='red', label='Comunic.'))
            lines_free.append([ax.plot([], [], color='blue', lw=2)[0] for _ in range(data['free_hist'][0].shape[0])])
            lines_comm.append([ax.plot([], [], color='red', lw=2)[0] for _ in range(data[variants[0]]['comm_config_hist'][0].shape[0])])
            ax.legend()

        # --- Métricas ---
        metric_ax.set_title("Evolución de la métrica")
        metric_ax.set_xlim(0, n_iter)
        metric_ax.set_xlabel("Iteración")
        metric_ax.set_ylabel("Valor")
        lines_metric = []
        for var in variants:
            line, = metric_ax.plot([], [], label=var.replace('_modelY','Y').replace('_modelG','G'))
            lines_metric.append((line, data[var]['MNF_hist']))
        metric_ax.legend(loc='lower left')

        
        def init():
            empty = np.empty((0,2))
            for s1, s2 in zip(sc_free, sc_comm):
                s1.set_offsets(empty)
                s2.set_offsets(empty)
            for tails_f, tails_c in zip(lines_free , lines_comm):
                for t in tails_f + tails_c:
                    t.set_data([], [])
            for line, _ in lines_metric:
                line.set_data([], [])
            return sc_free + sc_comm + [t for sub in lines_free for t in sub] + [t for sub in lines_comm for t in sub] + [l for l, _ in lines_metric]

        def update(frame):
            tail_len=5
            # free_pos es común a todas las variantes
            free_pos = data['free_hist'][frame]
            for j, var in enumerate(variants):
                comm_pos = data[var]['comm_config_hist'][frame]
                sc_free[j].set_offsets(free_pos)
                sc_comm[j].set_offsets(comm_pos)

                for k in range(free_pos.shape[0]):
                    hist_free = np.array([step[k] for step in data['free_hist'][max(0, frame+1-tail_len):frame + 1]])
                    lines_free[j][k].set_data(hist_free[:, 0], hist_free[:, 1])
                for k in range(comm_pos.shape[0]):
                    hist_comm = np.array([step[k] for step in data[var]['comm_config_hist'][max(0, frame+1-tail_len):frame + 1]])
                    lines_comm[j][k].set_data(hist_comm[:, 0], hist_comm[:, 1])
            
            x = np.arange(frame + 1)
            for line, metrica in lines_metric:
                line.set_data(x, metrica[:frame + 1])
            print(frame, len(lines_free), len(lines_comm), len(lines_metric))
            return sc_free + sc_comm + [t for sub in lines_free for t in sub] + [t for sub in lines_comm for t in sub] + [l for l, _ in lines_metric]

            
        ani = animation.FuncAnimation(fig, update, frames=n_iter, init_func=init, interval=200, blit=True)
        plt.tight_layout()

        if filename:
            base, ext = os.path.splitext(filename)
            exp_filename = f"{base}_exp{exp}{ext}"
            if ext==".gif":
                ani.save(exp_filename, writer='pillow', fps=fps)
            elif ext==".mp4":
                ani.save(exp_filename, writer='ffmpeg', fps=fps)
            else:
                raise ValueError("El archivo debe ser .mp4 o .gif")
            print(f"Animación guardada como '{exp_filename}'")
        plt.close(fig)


def plot_y_evolution(experiment_data, outputPath=None):
    modelos = experiment_data[0]['modelos'].keys()
    y_eval_permodel = {}
    for exp, data in enumerate(experiment_data):
        # Extraer listas de tensores (1,1)
        for model_name in modelos:
            aux = np.array([y for y in experiment_data[0]["modelos"][model_name]["modelY"]["y_evol"]])
            points = np.vstack([y for y in aux])  # shape (n,1)
            # Si querés shape (n,), aplastá:
            points = points.squeeze(-1)
            y_eval_permodel[model_name]=points
            #print(model_name, points.shape)

        # Graficar
        fig, ax = plt.subplots(1,len(modelos), figsize=(12,5))
        x = np.arange(len(y_eval_permodel[model_name]))

        #N_exp=len(log_data[0]['modelos']['real_5']['modelY']['y_evol'])
        n_pasos=len(experiment_data[0]['modelos'][model_name]['modelY']['y_evol'][0])

        for i, model_name in enumerate(modelos):
            ax[i].plot(y_eval_permodel[model_name])
            # --- Major ticks cada 5 ---
            ax[i].set_xticks(np.arange(0, len(x), n_pasos*10))
            # --- Minor ticks cada 1 ---
            ax[i].set_xticks(np.arange(0, len(x), n_pasos), minor=True)
            ax[i].set_title(model_name)

            # --- Grid major (oscura) ---
            ax[i].grid(
                True,
                which='major',
                axis='x',
                linestyle='--',
                linewidth=1.2,
                color='black',
                alpha=0.6
            )

            # --- Grid minor (clara) ---
            ax[i].grid(
                True,
                which='minor',
                axis='x',
                linestyle=':',
                linewidth=0.6,
                color='gray',
                alpha=0.3
            )

        plt.tight_layout()
        if outputPath is not None:
            plt.savefig(outputPath / f'y_evol_{exp}.pdf', dpi=300)
            plt.savefig(outputPath / f'y_evol_{exp}.png', dpi=300)
        else:
            plt.show()
        plt.close(fig)