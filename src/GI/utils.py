'''Utils.py

Misc utility functions
''' 
import numpy as np

def get_cumulative_data_indices(source_data_indices, test_frac=1.0):
    
    '''Returns cumulative indices
    
    for a list of lists [[x1,x2...],[y1,y2,..],[z1,z2,z3,...]] returns  [[x1,x2,..],[x1,x2,...,y1,y2,...],[x1,x2,..,y1,y2,..,z1,z2,z3...]]
    
    Arguments:
        source_data_indices {[type]} -- list of lists
        test_frac  -- if less than 1, this will split train test etc.
    
    '''
    try: 
        curr_indices = source_data_indices[0].tolist()[:int(test_frac*len(source_data_indices[0]))]
    except:
        curr_indices = source_data_indices[0][:int(test_frac*len(source_data_indices[0]))]
    # if curr_indices
    cumulative_indices = [[x for x in curr_indices]]
    for i in range(1,len(source_data_indices)):
        try:
            curr_indices = curr_indices + source_data_indices[i].tolist()[:int(test_frac*len(source_data_indices[i]))]
        except:
            curr_indices = curr_indices + source_data_indices[i][:int(test_frac*len(source_data_indices[i]))]
        cumulative_indices.append([x for x in curr_indices])
    return cumulative_indices

def store_numpy_array(filepath, array, allow_pickle=True, fix_imports=True):

    with open(filepath, 'wb') as file:
        np.save(file, array, allow_pickle, fix_imports)

def get_closest(lst,item):
    '''Returns the element of the list closest in value to item
    
    
    Arguments:
        lst {[type]} -- [description]
        item {[type]} -- [description]
    ''' 
    index = np.argmin(np.abs(np.array([x-item for x in lst])))
    return lst[index]