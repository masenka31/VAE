using Plots
using Flux
using Distributions
using LinearAlgebra
using LaTeXStrings
using IPMeasures

# include functions that generate new data
include(scriptsdir("WAE", "wae_functions.jl"))

# data creation
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

# network definition
nx = 2
nz = 2
nh = 5;
A, μ, logσ = Dense(nx, nh, swish), Dense(nh, nz), Dense(nh, nz)
f = Chain(Dense(nz, nh, swish), Dense(nh, nx))
z(μ, logσ) = μ .+ exp.(logσ) .* randn(size(μ))
KL(μ, logσ) = 0.5 * sum((exp.(logσ)).^2 .+ μ.^2 .- 1.0 .- 2. * logσ)
opt = Flux.ADAM(1e-2)
ps = Flux.params(A, μ, logσ, f)

# parameters and kernels
γ = 0.01
c = 1
kernel = GaussianKernel(γ)
# kernel = IMQKernel(c)
n_sample = 200

# loss function
s = 0.5
function loss(x)
    HID = A(x);
    zsample = z(μ(HID), logσ(HID))
    sample_rnd = hcat([randn(nz) for i in 1:n_sample]...)
    sample_z = hcat([z(μ(HID), logσ(HID)) for i in 1:n_sample]...)
    0.5 * sum((x .- f(zsample)).^2) .+ s * mmd(kernel, sample_rnd, sample_z)
end

# training
k = 1
@time for i in k:k + 49
    Flux.train!(loss, ps, data_train, opt)
    if i % 10 == 0
        println("Epoch: $i, loss: $(loss(dataT[1]))")
    end
    if isnan(loss(dataT[1]))
        println("Loss in NaN!")
        break
    end
end

# generate new data and reconstructed data
gen = generate_new(500, nz, f)
reco = reconstruct_data(dataT)

# plot
scatter(x,y,aspect_ratio=:equal,label=L"X");
plt_gen = scatter!(gen[1,:], gen[2,:], label=L"f(\varepsilon)")
scatter(x, y, aspect_ratio=:equal, label=L"X");
plt_rec = scatter!(reco[1,:], reco[2,:], label=L"\mathrm{VAE}(X)")