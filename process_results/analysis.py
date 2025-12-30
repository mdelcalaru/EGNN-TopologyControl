import numpy as np

def safe_relative(a, b):
    return (a - b) / (abs(b) + 1e-8)

def compute_relative_differences_static(log_data, ref_variant="maxL2", excluded_variants=None):
    
    if 'modelos' in log_data[0].keys():      
        resultados_lista = [
            {
                **{'ini': d['ini'], 'maxL2': d.get('maxL2', None), 'noSA': d.get('noSA', None)},
                **{f"{modelo}_{subkey}": valor
                for modelo, subdict in d['modelos'].items()
                for subkey, valor in subdict.items()}
            }
            for d in log_data
        ]
    variants = [k for k in resultados_lista[0] if k not in {'ini'}.union(excluded_variants or [])]
    print(variants)
    #print('Keys:', log_data[0].keys())
    results = {}
    for key in ['ini', ref_variant]:
        for variant_key in variants:
            if key != variant_key:
                results[f'mnf-{key}-{variant_key}'] = []
    #print(results)
    for variant_key in variants:
        if variant_key != ref_variant:
            results[f'time-{variant_key}'] = []        
    
    print('internal keys', results.keys())
    for i, data in enumerate(resultados_lista):
        if 'ini' not in data: 
            print(f'Muestra {i}: faltan datos iniciales')
            continue
        
        ref_ini = data['ini']['MNF']
        if data.get(ref_variant) is not None and data[ref_variant].get('MNF') is not None:
            ref_mnf = data[ref_variant]['MNF']
            ref_time = data[ref_variant].get('time')
            results[f'mnf-ini-{ref_variant}'].append(safe_relative(ref_mnf, ref_ini))
           
        # Referencia para comparaciones
        
        for variant_key in variants:
            if variant_key == ref_variant:
                continue    
            variant_data = data.get(variant_key)
            if variant_data is None or variant_data.get('MNF') is None:
                print(f"{variant_key} is {variant_data}")
                continue
            if variant_data.get('convergio') is False:
               print(f"[INFO] Experimento {i}: variante {variant_key} no convergió.")
               #continue
            # MNF relativo
            results[f'mnf-{ref_variant}-{variant_key}'].append(safe_relative(variant_data['MNF'], ref_mnf))
            # Tiempo relativo
            if variant_data.get('time') is None:
                continue
            if ref_time is not None:
                results[f'time-{variant_key}'].append(safe_relative(variant_data['time'], ref_time))
            else:
                results[f'time-{variant_key}'].append(variant_data['time'])
            # Respecto a inicial
            results[f'mnf-ini-{variant_key}'].append(safe_relative(variant_data['MNF'], ref_ini))
            

    results = {k: v for k, v in results.items() if len(v) > 0}
    # Convertir a arrays numpy (multiplicando por 100 para obtener %)
    for key in results:
        results[key] = 100 * np.array(results[key])

    return results


def compute_relative_differences_dynamic(log_data):
    if 'modelos' in log_data[0].keys():      
        resultados_lista = [
            {
                **{'ini': d['ini'], 'maxL2': d.get('maxL2', None), 'noSA': d.get('noSA', None)},
                **{f"{modelo}_{subkey}": valor
                for modelo, subdict in d['modelos'].items()
                for subkey, valor in subdict.items()}
            }
            for d in log_data
        ]
      
    results_per_variant = {
        key: [] for key in resultados_lista[0].keys() if (key != 'ini') 
    }
    print("Variants found:", results_per_variant.keys())

    for i, data in enumerate(resultados_lista):
        maxL2_data = data.get('maxL2')
        if maxL2_data is None or 'MNF_hist' not in maxL2_data:
            print(f"[WARN] Experimento {i}: faltan datos de maxL2. Comparando contra inicial.")
            results_per_variant.pop('maxL2', None)
            ref_mnf=np.array(data['ini']['MNF'])
        else:
            ref_mnf = np.array(maxL2_data['MNF_hist'])

        if len(ref_mnf.shape) != 1:
            ref_mnf = ref_mnf.flatten()

        if np.any(np.isnan(ref_mnf)) or ref_mnf.shape[0] == 0:
            print(f"[WARN] Experimento {i}: MNF de maxL2 inválido.")
#            continue

        for key in results_per_variant.keys():
            if key == 'maxL2':
                continue  # no tiene sentido comparar consigo mismo

            var_data = data.get(key)
            if var_data is None or 'MNF_hist' not in var_data:
                print(f"[INFO] Experimento {i}: variante {key} sin datos.")
                continue

            pred_mnf = np.array(var_data['MNF_hist']).flatten()

            if pred_mnf.shape != ref_mnf.shape:
                print(f"[WARN] Experimento {i}, variante {key}: tamaño incompatible ({pred_mnf.shape} vs {ref_mnf.shape})")
                continue

            rel_error = safe_relative(pred_mnf, ref_mnf)
            results_per_variant[key].append(100 * rel_error)  # en porcentaje

    return results_per_variant

def summarize_dynamic_results(results_per_variant):
    """
    Calcula media y desviación estándar por paso para cada variante.

    Args:
        results_per_variant: salida de compute_relative_differences_dynamic.

    Returns:
        Dict[variant] = (media, std) como arrays de longitud n_pasos.
    """
    summary = {}
    for variant, runs in results_per_variant.items():
        if not runs:
            continue
        data = np.stack(runs)  # shape: (n_experimentos, n_pasos)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        summary[variant] = (mean, std)
    return summary


