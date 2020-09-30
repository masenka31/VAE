This is a project using VAE for anomaly detection on HTRU2 dataset. This project is managed by `DrWatson`.

To run an experiment an experiment, simply change to this directory and do

```
using DrWatson
quickactivate("pulsars")
"""
best_script.jl runs experiment for VAE
wae_pulsars.jl runs experiment for WAE
change the script for different hyperparameters and network architectures
"""
include(scriptsdir("best_script.jl"))
# include(scriptsdir("WAE/wae_pulsars.jl"))
```

