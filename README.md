## Large Graph
This project aims to create a learning-based algorithm to automatically solve complex Tiling Problem.

### Requirements
+ Xvfb and xvfb-run
    Dependency xvfb is added for running qt code without real display. Usually simply installing the package with yum or apt is fine, while sometimes using slurm there might be shared lib not found issue.
    In this case we may have to copy the missing shared lib (found by `ldd $(which Xvfb)`) to an available place (say, home dir), and set LD_LIBRARY_PATH when running.
    My command submitted to slurm is:
    `LD_LIBRARY_PATH=/research/dept8/fyp21/cwf2101/rchuan/usr/lib64 xvfb-run python3 scripts/train_selector.py`
+ python==3.8
+ pytorch==1.9.1
+ torch_geometric==2.0.1
### Training
#### Train Solver Network
1. Generate complete graph and raw dataset
2. Edit or create config file, including "solver" and "tiling" parts
3. `xvfb-run python3 scripts/train_solver.py -c configs/yourconfig.json`
4. View the results by 
    + `tensorboard --logdir results --port xxxx`
    + ./results/yourlogpath/out.log

#### Train Selector Network
1. Generate complete graph and raw dataset 
2. Edit or create config file, including "selector" and "tiling" parts. "model_save_path", "loss" and "network" in  "Solver" part should also be configured in order to use pre-trained solver network.
3. `xvfb-run python3 script/train_selector.py -c configs/yourconfig.json` by default, and the training script can be run without xvfb if "show_intermediate" is set to false.
4. View the results by
    + `tensorboard --logdir results --port xxx`
    + ./results/yourlogpath/training.log
    + ./results/yourlogpath/all.log
    + images in ./results/yourlogpath/

### Evaluating
#### Evaluate Solver Network
1. Prepare contours and edit silhoutte list
2. `xvfb-run python3 scripts/draw.py`
#### Evaluate Selector Network
Not implemented yet
