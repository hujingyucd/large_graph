## Large Graph
This project aims to create a learning-based algorithm to automatically solve complex Tiling Problem.

Dependency xvfb is added for running qt code without real display. Usually simply installing the package with yum or apt is fine, while sometimes using slurm there might be shared lib not found issue. In this case we may have to copy the missing shared lib to an available place, and set LD_LIBRARY_PATH before running.
My script for submitting slurm jobs is:
LD_LIBRARY_PATH=/research/dept8/fyp21/cwf2101/rchuan/usr/lib64 xvfb-run python3 scripts/train_network.py

