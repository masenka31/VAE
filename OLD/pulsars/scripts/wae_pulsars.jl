# load data
include(scriptsdir("init_pulsars.jl"))
using IPMeasures

"""
model = BSON.load(datadir("ALL_DATA_WAE(IMQ)/run-hard-2_auc1=0.914_c=1_i=300_s=1_η=0.bson"))
@unpack i,A,μ,logs,f,opt,fpr,tpr,auc1,s,η,c = model
"""

# parameters
nx = 8
nz = 2
nh = 2;
γ = 0.01
c = 1
s = 1
n_sample = 100
ep = 300
η = 0.0001

# model 
opt = Flux.ADAM(η)
A, μ, logs = Dense(nx, nh, σ), Dense(nh, nz), Dense(nh, nz)
f = Chain(Dense(nz,nh,σ),Dense(nh,nx))
z(μ, logs) = μ .+ exp.(logs) .* randn(size(μ))
KL(μ, logs) = 0.5 * sum((exp.(logs)).^2 .+ μ.^2 .- 1.0 .- 2. *logs)
kernel = IMQKernel(c)
# kernel = GaussianKernel(γ)
ps = Flux.params(A, μ, logs, f)

# loss function 
function loss(x)
    HID = A(x);
    zsample = z(μ(HID),logs(HID))
    sample_rnd = hcat([randn(nz) for i in 1:n_sample]...)
    sample_z = hcat([z(μ(HID),logs(HID)) for i in 1:n_sample]...)
    0.5*sum((x.-f(zsample)).^2) .+ s*mmd(kernel,sample_rnd,sample_z)
end

rec_K = 50
function rec_loss(x)
    HID = A(x)
    sample = hcat([z(μ(HID), logs(HID)) for i in 1:rec_K]...)
    X = hcat([f(sample[:,i]) for i in 1:rec_K]...)
    Y = vcat([sum((x .- X[:,i]).^2) for i in 1:rec_K]...)
    mean(Y)
end

x = dataN_train[1]

# training procedure
@time for i in 1:ep
    Flux.train!(loss, ps, data_train, opt)
    global l = loss(x)
    if isnan(l)
        println("loss diverged into NaN, training terminated...")
        break
    end
    if i%5 == 0
        L_train = loss.(dataN)
        LA = loss.(dataA)
        avg_norm = mean(L_train)
        avg_an = mean(LA)
        L = rec_loss.(dataT)
        fpr, tpr = roccurve(L, labels)
        auc1 = auc(fpr,tpr)
        final_model = @dict(i,A,μ,logs,f,opt,fpr,tpr,auc1,s,η,c)
        safesave(datadir("ALL_DATA_WAE(IMQ)", savename("run-6", final_model, "bson")), final_model)
        println("Epoch: $i, l: $(round(l)), LN: $(round(avg_norm)), LA: $(round(avg_an))")
        println("AUC: $auc1")
    end
    println("Epoch: $i, l: $(round(l))")
end

printstyled("Current experiment completed.\n", bold = true, color = :cyan) 

