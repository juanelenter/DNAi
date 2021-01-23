#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""                  
                            dnai
    results.py:

      A class for fetching Comet experiment results

-._    _.--'"`'--._    _.--'"`'--._    _.--'"`'--._    _   
    '-:`.'|`|"':-.  '-:`.'|`|"':-.  '-:`.'|`|"':-.  '.` : '.   
  '.  '.  | |  | |'.  '.  | |  | |'.  '.  | |  | |'.  '.:   '.  '.
  : '.  '.| |  | |  '.  '.| |  | |  '.  '.| |  | |  '.  '.  : '.  `.
  '   '.  `.:_ | :_.' '.  `.:_ | :_.' '.  `.:_ | :_.' '.  `.'   `.
         `-..,..-'       `-..,..-'       `-..,..-'       `         `
"""
from comet_ml.api import API
import pandas as pd

class fetcher:
    '''
    An Interface for fetching experiment results through comet API.
    
    Example
    -------
    >>>  y = fetcher(project=["crossa-wheat-env-0", "crossa-wheat-env-1"])
    >>>  results = y.get_results(parameters = ["run_arg_9", "model", "coding"])
    '''
    
    def __init__(self, workspace="dna-i", project=None):
        '''
        Initializes API key and optionally which workspace and projects to use

        Parameters
        ----------
        workspace : STR, optional
            Comet workspace. The default is "dna-i".
        project : list(STR), optional
            List of Projects to use. The default is None.

        Returns
        -------
        None.

        '''
        self.workspace = workspace
        self.project = project
        self.api = API(api_key="V0OXnWOi4KVNS4OkwLjdnxSgK")
        
    def get_all_projects(self, workspace="dna-i"):
        '''
        Gets and prints all available projects.

        Parameters
        ----------
        workspace : STR, optional
            The default is "dna-i".

        Returns
        -------
        LIST(STR)
            available projects

        '''
        self.available_projects = self.api.get(workspace)
        print(self.available_projects)
        return self.available_projects
    
    def get_all_experiments(self, project=None):
        '''
        Fetches all experiments for a given project

        Parameters
        ----------
        project : STR, optional
             The default is None.

        Returns
        -------
        COMET EXPERIMENT OBJ

        '''
        if project is not None:
            self.experiments = self.api.get(f"{self.workspace}/{project}")
            print("fetching experiments")
        else:
            experiment = []            
            for proj in self.available_projects:
                elem = {'project': proj,
                        'experiments':self.api.get(f"{self.workspace}/{proj}")}
                experiment.append(elem)
            self.experiments = experiment
        return self.experiments
    
    def get_exp_keys(self):
        '''
        Converts list of comet experiment objects to experiment keys.
        
        Returns
        -------
        list of str

        '''
        self.keys = []
        print("fetching experiment keys")
        for exp in self.experiments:
            self.keys.append(exp.get_metadata()['experimentKey'])                
        return self.keys
    
    def result_to_df(self):
        '''
        Converts list of dictionaries result created with get_results to
        human friendly to a pandas dataframe.

        Returns
        -------
        DataFrame

        '''
        r = []
        for p in self.results:  
            result = p["metrics"]
            project = p["project"]
            for k,v in result.items():
                d = {}
                d["database"] = project
                d['experimentKey']=v["experimentKey"]
                d = {**d, **v['params']}
                for metric in v["metrics"]:
                    d['metricName']= metric["metricName"]
                    for value, step  in zip(metric["values"], metric["epochs"]):
                        aux= { **d, 'value': value,'step': step,}
                        r.append(aux)
            self.results = pd.DataFrame(r)
    
    def get_results(self, metrics=["r_test (r)", "mse_test", "r_train (r)",
                                   "mse_train", "r_test (r)_mean", "mse_test_mean"],
                    parameters=["model", "coding"], project=None):
        '''
        Fetches experiment metrics and parameters

        Parameters
        ----------
        metrics : LIST(STR), optional
            Which metrics to retrieve. The default is ["r_test (r)", "mse_test", "r_train (r)",                                   "mse_train", "r_test (r)_mean", "mse_test_mean"].
        parameters : LIST(STR), optional
            Which param to retrieve. The default is ["model", "coding"].
        project : LIST(STR), optional
            Which projects to retriever data from. The default is None.

        Returns
        -------
        Pandas dataframe with results
        '''
        print(f"Fetching {metrics} and {parameters}")
        if project is None:
            project = self.project
        self.results = []
        for proj in project:
            print(f"Project {proj}")
            self.get_all_experiments(project=proj)
            self.get_exp_keys()
            value = self.api.get_metrics_for_chart(self.keys, full=True, metrics = metrics, parameters=parameters)
            self.results.append({"project": proj, "metrics":value })
        self.result_to_df()
        return self.results


#y.get_all_results()
#%%
 
#y = fetcher(project=["crossa-wheat-env-0", "crossa-wheat-env-1"])
#ch = y.get_results(parameters = ["run_arg_9", "model", "coding"])


#%% 

      
        
    
        
        
