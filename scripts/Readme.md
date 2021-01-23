# Usage:

## Trayendo lo necesario
### Solo la primera vez

Primero clonarse dnai. Con los paths como los tengo configurados, *desde el home* correr:
```bash
git clone https://gitlab.fing.edu.uy/ihounie/dnai.git
```
Vamos a correr las cosas desde scripts
```bash
cd ./dnai/scripts/
```
Luego hay  que descargar la imagen de Docker (con singularity) para poder correr desde el contenedor,
que  tiene instalado todo lo necesario.

Como usa singularity pull, y singularity está instalado en los nodos de cómputo, hay que correrlo como un trabajo de slurm.
Pueden correr una sesion interactiva:

#### Run (interactivo)
```bash
interactivo -g
singularity pull docker://ihounie/dnai:r
```
También necesitan traerse las bases de datos que vayan a usar, para eso hay scripts en /data
(*En un futuro cercano preprocessing se va a encargar de descargarlas*)
Para usar unzip necesitamos correr desde un nodo, por ejemplo para descargar jersey:

#### Run (interactivo)
```bash
interactivo -g
cd ~/dnai/data
./get_data_jersey.sh
```

## Largando a correr

### Preprocessing 
El script que hay que correr es `sbatch_all_pre.sh`
```bash
cd ~/dnai/scripts/
./sbatch_all_pre.sh <number of fenos/envs> <base name> -f <format> -e <encoding> -i <imputation> -nan_flag <imputation flag> -ns  <nsplits>
```

Con el flag -h se despliegan las opciones:

```bash
cd ~/dnai/scripts/
sbatch_all_pre.sh -h
```
### Train Test

El script que hay que correr es `sbatch_all_post.sh`
que encola un trabajo donde  llama  a `10_splits_1_method_post.sh` para cada modelo y cada environment.

#### Batch
```bash
./sbatch_all_post.sh<n fenos/env> <nombre base> <model (all para correr todos)> <config_name sin el .json>  <-g> <-f> <-cg>
```
- `nfenos` es el número de ambientes o fenotipos
- Para el config no hace falta la ruta, sólo el nombre (e.g: `yeast_no_codif_del_rows_2020-07-17`)
- `nombre base` es para loggear en comet por defecto (e.g.: crossa-wheat se loguea como crossa-wheat-env-n por defecto)
- si pasas -cg carga el grid <nombre metodo>.json por defecto, no hace falta pasarle el nombre.
- model pasando all se corren todos, sino se puede pasar 1 modelo e.g.: svm
- Con el flag -h se despliegan las opciones

#### Interactivo
También pueden correrlo como un trabajo interactivo (aunque pueden ver la salida en comet en tiempo real también).


```bash
srun --job-name=dnai --time=12:00:00 --partition=normal --ntasks=1 --cpus-per-task=15 --mem=30G --pty bash -l
./10_splits_1_method_post.sh <args>
```

### Consultar estado de trabajos

```bash
squeue -u <nombre_de_usuario>
```

# Correr notebooks Interactivos desde el cluster

Se necesita correr desde una imagende singularity. Todas las de docker://ihounie/dnai tienen jupyter instalado.
Si vamos a usar Keras:

#### Run (interactivo)
```bash
interactivo -g
singularity pull docker://ihounie/dnai:keras
```

## Trabajo interactivo (con GPU)

```bash
srun --job-name=mitrabajo --time=24:00:00 --partition=normal --qos=gpu --gres=gpu:1 --ntasks=1 --mem=32G --cpus-per-task=20 --pty bash -l 
```
### Shell interactivo Singularity
Desde el nodo correr
```bash
singularity shell --nv dnai-keras.simg
```

### Jupyter notebook
Desde la imagen de singularity
```bash
jupyter notebook --ip=0.0.0.0 --port=8000 
```
### Túnel ssh
```bash
ssh -L <puerto_local>:node<n>.datos.cluster.uy:8000 <usuario>@cluster.uy
```



Y con eso creo  que lo que es estar,  estaríamos estando. Buen cluster para todos!