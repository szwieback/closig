'''
Created on Nov 22, 2023

@author: simon
'''
import pickle
import zlib
import numpy as np

def enforce_directory(path):
    try:
        path.mkdir(parents=False, exist_ok=True)
    except:
        raise
        
def save_object(obj, filename):
    enforce_directory(filename.parent)
    with open(filename, 'wb') as f:
        f.write(
            zlib.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)))

def load_C(fn):
    from greg import assemble_tril
    Cvec = np.moveaxis(np.load(fn), 0, -1)
    C = assemble_tril(Cvec, lower=False)
    return C  

def load_object(filename):
    if filename.suffix == '.npy':
        return np.load(filename)
    with open(filename, 'rb') as f:
        obj = pickle.loads(zlib.decompress(f.read()))
    return obj