import numpy as np 
from fetching import fetcher
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

def get_results(project, metrics = ["r_test"], params = ["model", "coding", "n_feno/env"], nfeno='all'):

    fet = fetcher(project = project)
    results = fet.get_results(metrics, params)
    
    r_test = results[results['metricName']=='r_test']
    r_test = r_test.replace("no_codif", "Additive")
    r_test = r_test.replace("ohe", "One Hot")
    r_test = r_test.replace("bayes_skl", "bayesian\n regression")
    r_test = r_test.rename(columns={'coding':'Coding'})
    
    return r_test

#%%
r_test = get_results(['yeast-final'], metrics = ["r_test"], params = ["model", "coding", "n_feno/env"], nfeno='all')

#%%
datos_grin = []
results = [0.80, 0.50, 0.53, 0.50, 0.52, 0.23, 0.71, 0.81, 0.61, 0.63, 0.23, 0.20]
for i in range(12):
    datos_grin.append({"model": " Grinberg\n et al.", "Coding": "Additive",'value': results[i], "n_feno/env":  str(i) })
r_test = r_test.append(datos_grin)
#%%
grouped = r_test.groupby('n_feno/env')
env = ["Cadmium Chloride", 'Congo red', 'Cycloheximide', 'Diamide', 'Ethanol', 'Hydroquinone', 'Lithium Chloride', 'Maltose', 'Neomycin', 'Tunicamycin', "Galactose", "YNB:ph3"]
sns.set(font_scale=1)
for name, group in grouped:
    group = group.sort_values(by ='model' )
    sns.set_style("whitegrid")
    i=0
    #plt.figure(figsize=(25,10))
    ax = sns.boxplot(data=group, x="model", y="value",  palette="pastel", dodge=True)
    sns.swarmplot(data=group, x="model", y="value", palette="deep",dodge=True, ax=ax)
    ax.set(xlabel='Model', ylabel='Predictive Correlation')
    plt.title("{}".format(env[int(name)]))
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.savefig( "yeast_env{}".format(name), transparent=True, dpi=500)
    plt.show()
