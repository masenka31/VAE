using Plots
using Flux
using Distributions
using LinearAlgebra
using LaTeXStrings

include(scriptsdir("anomaly_detection", "emvae_functions.jl")

# data
ndat = 500
X = rand(MvNormal([3,5],I),ndat)
x = X[1,:]
y = X[2,:]
"""
First transformation:
x = 2 .* x .+ y
y = x + y
Second transformation:
x = 0.5 .* x .+ y .^2
y = 2 .* y
Two gaussians:
Y = rand(MvNormal(zeros(2),I),800)
Z = rand(MvNormal([0;10],I),500)
X = hcat(X,Y)
x = X[1,:]
y = X[2,:]
"""
scatter(x,y,aspect_ratio=:equal)

dataT = [ vcat(x[i], y[i]) for i in 1:ndat]
data_train = zip(dataT,)
opt = Flux.ADAM(1e-2);

# define model
nx = 2
nz = 2
nh = 2;
A, μ, logσ = Dense(nx, nh, swish), Dense(nh, nz), Dense(nh, nz);
f = Chain(Dense(nz,nh,swish),Dense(nh,nx));
z(μ, logσ) = μ .+ exp.(logσ) .* randn(size(μ));
KL(μ, logσ) = 0.5 * sum((exp.(logσ)).^2 .+ μ.^2 .- 1.0 .- 2. *logσ);
ps = Flux.params(A, μ, logσ, f);

# parameter s and loss function
s = 0.005;
function loss(x)
    HID = A(x);
    zsample = z(μ(HID),logσ(HID));
    0.5*sum((x.-f(zsample)).^2) + s*KL(μ(HID),logσ(HID))
end

# training
k = 1
for i in k:k+99
    Flux.train!(loss,ps,data_train,opt)
    prumer = round(sum(loss.(dataT))/ndat,digits=4)
    max = round(maximum(loss.(dataT)),digits=4)
    println("Epoch: $i, avg=$prumer, max=$max")
end
k = k + 100

# function to generate new data from the model
function generate_new(f,nz)
    latent = rand(MvNormal(zeros(nz),I),n)
    new = hcat([f(latent[:,j]) for j in 1:n]...)
end

# plot training and new data
scatter(x,y,aspect_ratio=:equal,label=L"X")
gen = generate_new(f,nz)
scatter(gen[1,:],gen[2,:],label=L"f(\varepsilon)")