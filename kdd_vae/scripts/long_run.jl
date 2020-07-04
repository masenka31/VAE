cd("/mnt/lustre/helios-home/maskomic/Desktop/VAE/kdd_vae")
using Pkg
Pkg.add("DrWatson")
using DrWatson
@quickactivate("kdd_vae")
include(scriptsdir("init_kdd.jl"))
include(scriptsdir("best_script.jl"))