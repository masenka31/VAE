# Gaussian Mixture
using Distributions
using LinearAlgebra
using Plots
using Flux
using IPMeasures

function rand_mixture(m1,m2,p)
    a = rand(Multinomial(1,p))[1] 
    if a == 0
        return rand(m1)
    else
        return rand(m2)
    end
end

function loss(x)
    HID = A(x);
    zsample = z(μ(HID),logσ(HID))
    sample_rnd = hcat([randn(nz) for i in 1:n_sample]...)
    sample_z = hcat([z(μ(HID),logσ(HID)) for i in 1:n_sample]...)
    0.5*sum((x.-f(zsample)).^2) .+ s*mmd(kernel,sample_rnd,sample_z)
end

function encoding(x)
    HID = A(x)
    zs = z(μ(HID),logσ(HID))
end

function data_encoding(dataT)
    return hcat([encoding(dataT[i]) for i in 1:ndat]...)
end

function generate_new(n,nz,f)
    zs = rand(MvNormal(zeros(nz),I),n)
    new = Float64.(hcat([f(zs[:,i]) for i in 1:n]...))
    return new
end

function reconstruct(x)
    HID = A(x);
    zsample = z(μ(HID),logσ(HID))
    return f(zsample)
end

function reconstruct_data(dataT)
    return hcat([reconstruct(dataT[i]) for i in 1:ndat]...)
end


m1 = MvNormal(zeros(2),I)
m2 = MvNormal([0;10],I)

ndat = 1000
p = [0.4;0.6] 
X = hcat([rand_mixture(m1,m2,p) for i in 1:ndat]...)
x = X[1,:]
y = X[2,:]
scatter(x,y,aspect_ratio=:equal)

dataT = [[x[i] y[i]]' for i in 1:ndat]
data_train = zip(dataT,) |> gpu   

nx = 2
nz = 2
nh = 5;
A, μ, logσ = Dense(nx, nh), Dense(nh, nz), Dense(nh, nz) |> gpu  
f = Chain(Dense(nz,nh),Dense(nh,nx)) |> gpu  
z(μ, logσ) = μ .+ exp.(logσ) .* randn(size(μ))
KL(μ, logσ) = 0.5 * sum((exp.(logσ)).^2 .+ μ.^2 .- 1.0 .- 2. *logσ)
γ = 0.01
kernel = GaussianKernel(γ) #IMQKernel()  
n_sample = 400
opt = Flux.ADAM(1e-2)
ps = Flux.params(A, μ, logσ, f)

s = 0.1
p = [0.5;0.5] 

function loss(x,sample_rnd)
    HID = A(x);
    zsample = z(μ(HID),logσ(HID))
    sample_z = hcat([z(μ(HID),logσ(HID)) for i in 1:n_sample]...)
    sample = sample_rnd
    0.5*sum((x.-f(zsample)).^2) .+ s*mmd(kernel,sample,sample_z)
end

k = 1
@time for i in k:k+4
    sample_rnd = [hcat([rand_mixture(m1,m2,p) for i in 1:n_sample]...) for i in 1:ndat] 
    Flux.train!(loss,ps,zip(dataT,sample_rnd),opt)
    println("Epoch: $i, loss: $(loss(dataT[1],sample_rnd[1]))")
    if isnan(loss(dataT[1],sample_rnd[1]))
        println("Loss in NaN!")
        break
    end
end
k = k+5

function new_sample(nz,f,m1,m2,p)
    a = rand_mixture(m1,m2,p)
    return f(a)
end

function new_data(n,nz,f,m1,m2,p)
    new = hcat([new_sample(nz,f,m1,m2,p) for i in 1:n]...)
    return new
end

test = new_data(500, nz, f,m1,m2,p)
rec = reconstruct_data(dataT)
enc = data_encoding(dataT)
# scatter(enc[1,:],enc[2,:],aspect_ratio=:equal)

scatter(x, y, aspect_ratio = :equal, label = "X");
plt = scatter!(test[1,:], test[2,:], label = "f(e)");
scatter(x, y, aspect_ratio = :equal, label = "X");
plt2 = scatter!(rec[1,:], rec[2,:], label = "VAE(X)");

savefig(plt,"one.pdf")
savefig(plt2,"two.pdf")