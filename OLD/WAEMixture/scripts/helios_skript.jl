# activate environment
using Distributions
using LinearAlgebra
using Plots
using Flux
using IPMeasures

# functions 
include(scriptsdir("waem_functions.jl"))

# ;qsub -I -q student -l walltime=1:00:00 -l mem=60G,ncpus=8

function loss(x,sample_rnd)
    HID = A(x);
    zsample = z(μ(HID),logσ(HID))
    sample_z = hcat([z(μ(HID),logσ(HID)) for i in 1:n_sample]...)
    sample = sample_rnd
    0.5*sum((x.-f(zsample)).^2) .+ s*mmd(kernel,sample,sample_z)
end

# latent space
p_Z() = rand_mixture(MvNormal([0;0],I),MvNormal([0;6],I),[0.5;0.5])
S = hcat([p_Z() for i in 1:10000]...)
# scatter(S[1,:],S[2,:],aspect_ratio=:equal,label="p(Z)")

# data generation
ndat = 500
p = [0.4;0.6]
m1 = MvNormal([0;0],I)
m2 = MvNormal([0;10],I) 
X = hcat([rand_mixture(m1,m2,p) for i in 1:ndat]...)
x = X[1,:]
y = X[2,:]
# scatter(x,y,aspect_ratio=:equal,label="X")

dataT = [[x[i] y[i]]' for i in 1:ndat]
data_train = zip(dataT,)

nx = 2
nz = 2
nh = 5;
A, μ, logσ = Dense(nx, nh), Dense(nh, nz), Dense(nh, nz)
f = Chain(Dense(nz,nh),Dense(nh,nx))
z(μ, logσ) = μ .+ exp.(logσ) .* randn(size(μ))
KL(μ, logσ) = 0.5 * sum((exp.(logσ)).^2 .+ μ.^2 .- 1.0 .- 2. *logσ)
γ = 0.01
kernel = GaussianKernel(γ) #IMQKernel()  
n_sample = 300
opt = Flux.ADAM(1e-2)
ps = Flux.params(A, μ, logσ, f)
s = 0.1

k = 1
@time for i in k:k+14
    sample_rnd = [hcat([p_Z() for i in 1:n_sample]...) for i in 1:ndat]
    Flux.train!(loss,ps,zip(dataT2,sample_rnd),opt)
    println("Epoch: $i, loss: $(loss(dataT[1],sample_rnd[1]))")
    if isnan(loss(dataT[1],sample_rnd[1]))
        println("Loss in NaN!")
        break
    end
end
k = k+15

test = new_data(500, nz, f,m1,m2,p)
rec = reconstruct_data(dataT)
enc = data_encoding(dataT)
plt_enc = scatter(enc[1,:],enc[2,:],aspect_ratio=:equal,label=L"p(Z|X)");

scatter(x, y, aspect_ratio = :equal, label = "X");
plt = scatter!(test[1,:], test[2,:], label = "f(e)");
scatter(x, y, aspect_ratio = :equal, label = "X");
plt2 = scatter!(rec[1,:], rec[2,:], label = "VAE(X)");

safesave(plotsdir(savename("enc",plt_enc,"pdf")),plt_enc)
safesave(plotsdir(savename("gen",plt,"pdf")),plt)
safesave(plotsdir(savename("rec",plt2,"pdf")),plt2)
