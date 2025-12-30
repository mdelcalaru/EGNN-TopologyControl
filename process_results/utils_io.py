import pickle as pk
import glob
import os

def load_log_file(file_path):

    with open(file_path, 'rb') as f:
        log_data = pk.load(f)
        if 'metadata' in log_data.keys():
            meta= log_data['metadata']
        if 'estadistica' in log_data.keys():
            log_data = log_data['estadistica']
            print(f"Loaded 'estadistica' from {file_path}, which contains {len(log_data)} entries.")
        if isinstance(log_data, dict) and all(isinstance(k, int) for k in log_data.keys()):
            log_data = [log_data[k] for k in sorted(log_data.keys())]
        elif not isinstance(log_data, list):
            print(f"Unexpected data format in {file_path}. Expected a list or dict with int keys.")
            return []
    return log_data, meta if 'meta' in locals() else None

def load_log_files(files):
    
    log_data = []
    for file in files:      
        with open(file, 'rb') as f:
            data=pk.load(f)
            if isinstance(data['estadistica'], list):
                log_data.extend(data['estadistica'])
            else:
                log_data.extend([data['estadistica']])
    
    if not log_data:
        print(f"Not valid files in {files}.")
    else:
        print(f"Loaded {len(log_data)} data files.")
        print(f"Example keys: {log_data[0].keys()}")
    return log_data

def validate_keys(data, required_keys):
    for key in required_keys:
        if key not in data:
            print(f"Missing key {key}")
            return False
    return True
