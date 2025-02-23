import numpy as np 


def output_format(data=None, lb=None, ub=None, bool_result=None, p_value=None, r2=None, txt=None, h_stat=None):
        #Parameters 
        #lb Lower Bound (Float, optional)
        #ub Upper Bound (Float, optional)
        #bool_result Boolean result (bool, optional)
        #r2 R^2 (float, optional)
    
    if data is not None:
        return data 
    
    output = {}


    if lb is not None and ub is not None:
        output['lb'] = lb
        output['ub'] = ub,

    if p_value is not None : 
        output['p_value'] = p_value

    if bool_result is not None:
        output['Result'] = np.bool_(bool_result)
    
    if r2 is not None: 
        output['R^2'] = r2
        
    if txt is not None:
        output['txt'] = txt   

    if h_stat is not None:
        output['H_statistic'] = h_stat

    return output
        
        

