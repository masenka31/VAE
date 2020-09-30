# load data and packages
include(scriptsdir("anomaly_detection", "init_pulsars.jl"))
using IPMeasures

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

# choose kernel
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
    # every 5 epochs count mean loss for normal data and anomaly data and save the model
    if i%5 == 0
        L_train = loss.(dataN)
        LA = loss.(dataA)
        avg_norm = mean(L_train)
        avg_an = mean(LA)
        L = rec_loss.(dataT)
        fpr, tpr = roccurve(L, labels)
        auc1 = auc(fpr,tpr)
        final_model = @dict(i,A,μ,logs,f,opt,fpr,tpr,auc1,s,η,c)
        safesave(datadir("WAE_pulsars", savename("run-1", final_model, "bson")), final_model)
        println("Epoch: $i, l: $(round(l)), LN: $(round(avg_norm)), LA: $(round(avg_an))")
        println("AUC: $auc1")
    end
    println("Epoch: $i, l: $(round(l))")
end

printstyled("Current experiment completed.\n", bold = true, color = :cyan) 

