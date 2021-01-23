# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:08:57 2020

@author: Tecla
"""
import os.path

data_path = os.path.abspath(os.path.join(os.pardir,"data"))
snps_path = os.path.abspath(os.path.join(data_path,"snps.txt"))
output_path = os.path.abspath(os.path.join(os.pardir,"output"))
sim_output_path = os.path.abspath(os.path.join(output_path,"sim"))
eval_output_path = os.path.abspath(os.path.join(output_path,"eval"))
snps_matrix_path =os.path.abspath(os.path.join(sim_output_path, "snps_matrix.csv")) 
phen_sim_output_path =os.path.abspath(os.path.join(sim_output_path, "phenotype_sim.csv"))
idx_train_output_path = os.path.abspath(os.path.join(sim_output_path, "idx_train.csv"))
idx_test_output_path = os.path.abspath(os.path.join(sim_output_path, "idx_test.csv"))
predictions_output_path = os.path.abspath(os.path.join(output_path,"pred"))
bayes_output_path =os.path.abspath(os.path.join(predictions_output_path,"bayes"))
bayes_b_output_path =os.path.abspath(os.path.join(bayes_output_path,"b"))
bayes_c_output_path =os.path.abspath(os.path.join(bayes_output_path,"c"))
ridge_output_path =os.path.abspath(os.path.join(predictions_output_path,"ridge"))
bglr_c_path = os.path.abspath(os.path.join(bayes_output_path, "y_bc.csv"))
bglr_b_bath = os.path.abspath(os.path.join(bayes_output_path, "y_bb.csv"))

#y_train_output_path =os.path.abspath(os.path.join(os.pardir, "output/y_train.csv"))
#y_test_output_path =os.path.abspath(os.path.join(os.pardir, "output/y_test.csv"))
#X_train_output_path =os.path.abspath(os.path.join(os.pardir, "output/X_train.csv"))
#X_test_output_path =os.path.abspath(os.path.join(os.pardir, "output/X_test.csv"))