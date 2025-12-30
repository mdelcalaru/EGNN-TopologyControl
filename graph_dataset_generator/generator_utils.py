import pickle as pk
from pathlib import Path
import datetime


def save_data_list(data_list, path):
    path = Path(path)
    if path.suffix:  # Tiene extensión, es archivo
        path.parent.mkdir(parents=True, exist_ok=True)
        fname = path
    else:  # Es directorio
        path.mkdir(parents=True, exist_ok=True)
        fname = path / "dataset.pt"
    with open(fname, 'wb') as f:
        pk.dump(data_list, f)
    
    

def generate_filename(out_dir, total_agents, epsilon, num_samples, split, timestamp=None, delaunay=False):
    # timestamp en formato YYYYMMDD_HHMMSS
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")   
    # Generar nombre de archivo con los parámetros
    if delaunay:
        filename = f"{timestamp}_agents{total_agents}_eps{str(epsilon).replace('.', '_')}_samples{num_samples}_delaunay_{split}.pt"
    else:
        filename = f"{timestamp}_agents{total_agents}_eps{str(epsilon).replace('.', '_')}_samples{num_samples}_{split}.pt"
    return Path(out_dir) / filename
