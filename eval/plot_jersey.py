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
r_test = get_results(['jersey-final', 'jersey2'], metrics = ["r_test"], params = ["model", "coding", "n_feno/env"], nfeno='all')

#%%
datos_grin = []
results = [0.30]
for i in range(len(results)):
    datos_grin.append({"model": " Razende\n et. al", "Coding": "Additive",'value': results[i], "n_feno/env":  str(i)})
r_test = r_test.append(datos_grin)
#%%
sns.set(font_scale=1)
group = r_test.sort_values(by ='model' )
sns.set_style("whitegrid")
#plt.figure(figsize=(25,10))
ax = sns.boxplot(data=group, x="model", y="value", hue ='Coding', palette="pastel", dodge=True)
sns.swarmplot(data=group, x="model", y="value",hue ='Coding',  palette="deep",dodge=True, ax=ax)
ax.set(xlabel='Model', ylabel='Predictive Correlation')
plt.title("SCR")
plt.gcf().subplots_adjust(bottom=0.25)
plt.savefig( "jersey", transparent=True,dpi=500 )
plt.show()
