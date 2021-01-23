import numpy as np 
from fetching import fetcher
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


def plot_results(project, metrics = ["r_test", "r_train"], params = ["model", "coding", "n_feno/env"], nfeno='all'):

    fet = fetcher(project = project)
    results = fet.get_results(metrics, params)
    
    r_test = results[results['metricName']=='r_test']
    sns.scatterplot(data=r_test, x="model", y="value", hue="coding")
    

#plot_results(["crossa-wheat-final"])
metrics = ["r_test", "r_train"]
params = ["model", "coding", "n_feno/env"]
fet = fetcher(project = ["crossa-wheat-final"])
results = fet.get_results(metrics, params)
r_test = results[results['metricName']=='r_test']
datos_crossa = []
baseline = [0.608,0.497, 0.478,  0.524]
r_test = r_test.replace("no_codif", "Additive")
r_test = r_test.replace("ohe", "One Hot")
for i in range(4):
    datos_crossa.append({"model": "Crossa et al.", "coding": "Additive",'value': baseline[i], "n_feno/env":  str(i) })
r_test = r_test.append(datos_crossa)
grouped = r_test.groupby("n_feno/env")
#%%
sns.set_style("whitegrid")
i=0
for name, group in grouped:
    #fig, ax = plt.subplots(figsize=(10,6))
    ax = sns.boxplot(data=group, x="model", y="value", hue="coding", palette="pastel", dodge=True)
    sns.swarmplot(data=group, x="model", y="value", hue="coding", palette="deep",dodge=True, ax=ax)
    ax.set(xlabel='Model', ylabel='Predictive Correlation')
    plt.title("Environment "+name)
    if i < (len(grouped)-1):
        plt.legend()
        i+=1
    plt.savefig( "Crossa_overall_env_"+name+'.png', transparent=True)
    plt.show()

