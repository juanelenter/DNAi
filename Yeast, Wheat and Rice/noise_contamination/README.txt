En la carpetas std, drop en indiv se encuentran los scripts para correr las pruebas. Std corresponde a la
contaminacion con ruido del fenotipo, drop a la pérdida de snps en el genotipo e indiv a la pérdida de individuos (samples).

Los nombres de cada script indican a qué modelo corresponden. En la carpeta results/{modelo} 
se guardan los resultados para cada ambiente utilizando {modelo}. La carpeta params guarda los 
parametros que encontramos con gridsearch para gbm y rf (faltaría svm a la fecha). Todos 
los paths son relativos al path donde esta el script, por lo que  hay que correrlos desde el 
directorio en el que se encuentran. Salute