# DNAi
# Loggear experimentos:

## Instalation
Hay que instalarse neptune y comet:
```bash
pip install neptune-client
pip install psutil
pip install comet_ml

```
Despues hay que llamar a train-test con -log_exp nombre_del_proyecto
(por ahora en ambos tengo un proyecto que se llama debug,
 desp la idea es que se loguee a un proyecto distinto segun la base de datos que estas usando)

# Bitácora: 
#### 22/4:
### Orden del día:

* Gianola puede predecir el ruido? r2>>h2
* Probamo CNN en giano con sus parametros, overfittin de aquellos
* seguimo con grinberg: Agregamos ruido
* Se nos cae otro idolo: Grinberg, scaleaste con todos los datos!?
* ClusterUY!
* Sudo en shannon?


#### 15/4:
* Terminar de replicar experimentos del paper de pancho/gianola, resultados
* Empezar a replicar la síntesis de Grinberg sobre fenotipos reales: Agregar ruido, borrar snps, generar epistasis
* Sistematizar, todos los errores/resultados tienen que plotearse para varios (eg:5) experimentos , i.e. train/test splits

### requirements
`Python3`,`numpy`, `sklearn`, `R`, `BGLR`

## Demo
* run `demo.sh`

## Non additive phenotype simulator

Implementation of non additive phenotype simulations
as described in
> Deep learning versus parametric and ensemble methods for genomic prediction of complex phenotypes
Abdollahi-Arpanahi et al. (2020)

[link](https://gsejournal.biomedcentral.com/articles/10.1186/s12711-020-00531-z)

#### Instructions:

* run *load.py* to load data
* run *non_additive_phenotype_simulator.py*

Simulated Phenotypes are stored in output. 
Ridge Regression results are  printed.

## R
*R scripts for benchmarking*

### Requirements:
* `BGLR`
* `truncnorm`

#### Instructions:
(Run from R folder, TODO: change paths to run from rootdir) 
* Rscript ./Predict_bayes.R 

## Data
*Please upload raw data only, not intermediate results*

* snps.txt contains 60000 SNPs for 1000 individuals


## Output

Simulation and prediction results should be directed here


## Util
 
Utilities for genome conversion

### requirements
`plint`

*TODO*: cleanup this folder

