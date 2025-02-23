import numpy as np 


def output_format(lb=None, ub=None, bool_result=None, p_value=None, r2=None, txt=None):
        #Parameters 
        #lb Lower Bound (Float, optional)
        #ub Upper Bound (Float, optional)
        #bool_result Boolean result (bool, optional)
        #r2 R^2 (float, optional)
    output = {}

    if lb is not None and ub is not None:
        output['lb'] = lb
        output['ub'] = ub

    if p_value is not None : 
        output['p_value'] = p_value

    if bool_result is not None:
        output['Result'] = np.bool_(bool_result)
    
    if r2 is not None: 
        output['R^2'] = r2
        
    if txt is not None:
        output['txt'] = txt         
    
    return output
        
        
def distribution_visualization(data, title=None, x_label=None, y_label=None, bins=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if bins is None:
        #buscar formula correcta para bins
        
        bins = np.log10(len(data))
####add a scatterplot , histogram and boxplot and table of contents

    plt.figure(figsize=(20, 10))
    ax1 = plt.subplots(212)
    ax1 = sns.scatterplot(data=data)
    ax1 = plt.title('Scatterplot of ' + title)
    ax1 = plt.xlabel(x_label)
    ax1 = plt.ylabel(y_label)
    
#### Histogram
    ax2 = plt.subplots(221)
    ax2 = sns.histplot(data, bins=bins, kde=True)
    ax2 = plt.title('Histogram of ' + title)
    ax2 = plt.xlabel(x_label)
    ax2 = plt.ylabel(y_label)

##########
#Boxplot     
    ax3= plt.subplots(222)
    ax3= sns.boxplot(data)
    ax3 = plt.title('Boxplot of ' + title)
    ax3 = plt.xlabel(x_label)
    ax3 = plt.ylabel(y_label)

    return plt.show()
