import numpy as np 


def output_format(ll=None, ul=None, bool_result=None, p_value=None, r2=None, txt=None):
        #Parameters 
        #LL Lower Limit (Float, optional)
        #UL Upper Limit (Float, optional)
        #bool_result Boolean result (bool, optional)
        #r2 R^2 (float, optional)
    output = {}

    if ll is not None and ul is not None:
        output['LL'] = ll
        output['UL'] = ul

    if p_value is not None : 
        output['p_value'] = p_value

    if bool_result is not None:
        output['Result'] = np.bool_(bool_result)
    
    if r2 is not None: 
        output['R^2'] = r2
        
    if txt is not None:
        output['txt'] = txt         
    
    return output
        
        
def positive_values_test(data):
    return np.all(np.array(data) >0 )

def to_numpy_array(data):
    """Convierte datos en un arreglo numpy."""
    return np.array(data, dtype=float)

def validate_non_nan(data):
    """Valida que no haya valores NaN o infinitos en los datos."""
    return np.all(np.isfinite(data))
