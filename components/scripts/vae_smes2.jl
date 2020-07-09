using Plots
using Distributions
using Flux
using LinearAlgebra
using IPMeasures
using LaTeXStrings

include("C:\\Users\\masen\\OneDrive\\SCHOOL\\BP\\programming\\funkce.jl")
cd("C:\\Users\\masen\\OneDrive\\SCHOOL\\BP\\programming\\smes_autoencoderu")

function generate_from_X(data, f, z, μ, logσ, A)
    new = zeros(nx, length(data))
    for i in 1:length(data)
        HID = A(data[i])
        zsample = z(μ(HID), logσ(HID))
        point = f(zsample)
        new[:,i] = point
    end
    return new
end

n1, n2 = 800, 500
ndat = n1 + n2
X1 = rand(MvNormal([0;0], I), n1)
X2 = rand(MvNormal([0;10], I), n2)
# x1 = 3 .* X1[2,:] .^2 .+ 0.2 .* randn(n1)
# y1 = 3 .* X1[2,:] .+ 0.2 .* randn(n1)
# X1 = vcat(x1',y1')
# x2 = - 8 .* X2[2,:] .^2 .+ 0.2 .* randn(n2) .+ 10
# y2 = 5 .* X2[2,:] .+ 0.2 .* randn(n2) .- 4
# X2 = vcat(x2',y2')
X = hcat(X1, X2)
scatter(X1[1,:],X1[2,:],aspect_ratio = :equal,label = L"X_1");# ,size=(400,400))
scatter!(X2[1,:],X2[2,:],aspect_ratio = :equal,label = L"X_2")

scatter(X[1,:],X[2,:],label = L"\mathrm{data}",aspect_ratio = :equal)
scatter(X[1,:],X[2,:],label = "X",aspect_ratio = :equal)

dataT = [ vcat(X[1,i], X[2,i]) for i in 1:ndat]
dataX1 = [ vcat(X1[1,i], X1[2,i]) for i in 1:n1]
dataX2 = [ vcat(X2[1,i], X2[2,i]) for i in 1:n2]
data_train = zip(dataT, )
labels = vcat(ones(n1), zeros(n2))
opt = Flux.ADAM(1e-2)

nx = 2
nz = 2
nh = 2;
z(μ, logσ) = μ .+ exp.(logσ) .* randn(size(μ))
A1, μ1, logσ1 = Dense(nx, nh, swish), Dense(nh, nz), Dense(nh, nz)
f1 = Chain(Dense(nz, nh, swish), Dense(nh, nx))
z1(μ1, logσ1) = μ1 .+ exp.(logσ1) .* randn(size(μ1))
A2, μ2, logσ2 = Dense(nx, nh, swish), Dense(nh, nz), Dense(nh, nz)
f2 = Chain(Dense(nz, nh, swish), Dense(nh, nx))
z2(μ2, logσ2) = μ2 .+ exp.(logσ2) .* randn(size(μ2))
ps = Flux.params(A1, μ1, logσ1, A2, μ2, logσ2, f1, f2);
KL(μ, logσ) = 0.5 * sum((exp.(logσ)).^2 .+ μ.^2 .- 1.0 .- 2. * logσ)

exponent(x,zs,f) = -sum((x .- f(zs)).^2)

function lik(x)
    HID1 = A1(x)
    zs1 = z1(μ1(HID1), logσ1(HID1))
    HID2 = A2(x)
    zs2 = z2(μ2(HID2), logσ2(HID2))
    exp1 = exponent(x, zs1, f1)
    exp2 = exponent(x, zs2, f2)
    max_exp = maximum([exp1,exp2])
    l_1 = exp(exp1 - max_exp) * α1
    l_2 = exp(exp2 - max_exp) * α2
    ss = l_1 + l_2
    l1 = l_1 / ss
    l2 = l_2 / ss
    return l1, l2
end
function lik_n(dataT)
    LI_1 = zeros(ndat)
    LI_2 = zeros(ndat)
    for i in 1:ndat
        L1, L2 = lik(dataT[i])
        LI_1[i] = L1
        LI_2[i] = L2
    end
    return LI_1, LI_2
end
function rec1(x)
    HID = A1(x)
    zsample = z1(μ1(HID), logσ1(HID))
    return 0.5 * b * sum((x .- f1(zsample)).^2)
end
function rec2(x)
    HID = A2(x)
    zsample = z2(μ2(HID), logσ2(HID))
    return 0.5 * b * sum((x .- f2(zsample)).^2)
end
function rec1(x)
    HID = A1(x)
    zsample = z(μ1(HID), logσ1(HID))
    return 0.5 * b * sum((x .- f1(zsample)).^2)
end
function rec2(x)
    HID = A2(x)
    zsample = z(μ2(HID), logσ2(HID))
    return 0.5 * b * sum((x .- f2(zsample)).^2)
end

function loss(x)
    li_1, li_2 = lik(x)
    li_1 * (rec1(x) + KL(μ1(A1(x)), logσ1(A1(x))) - log(α1 + 10^(-10))) + li_2 * (rec2(x) + KL(μ2(A2(x)), logσ2(A2(x))) - log(α2 + 10^(-10)))
end

α1, α2 = 0.5, 0.5
s = 0.06
b = 1 / s
k = 1

for i in k:k + 19
        # tady spočítám lik
    global LI1, LI2 = lik_n(dataT)
        # tady spočítám α
    global α1 = sum(LI1) / ndat
    global α2 = sum(LI2) / ndat
    Flux.train!(loss, ps, zip(dataT, ), opt)
    prumer = round(sum(loss.(dataT)) / ndat, digits = 4)
    max = round(maximum(loss.(dataT)), digits = 4)
    vr = round.([sum(LI1),sum(LI2)])
    if isnan(prumer)
        println("NaN")
        break
    end
    println("Epoch $i: n1=$vr, avg=$prumer, max=$max")
end
k = k + 20

LI1, LI2 = lik_n(dataT)
scatter(X[1,:],X[2,:],zcolor = LI1,aspect_ratio = :equal,label = "data classes")

A = generate_data(1000, nz, 2, f1)';
B = generate_data(1000, nz, 2, f2)';

scatter(X[1,:],X[2,:],label = L"\mathrm{data \;} X",aspect_ratio = :equal,legend = :topright);
scatter!(A[1,:],A[2,:],label = L"\mathrm{generated \; from \;} f_1(\varepsilon)");
scatter!(B[1,:],B[2,:],label = L"\mathrm{generated \; from \;} f_2(\varepsilon)")

scatter(X[1,:],X[2,:],label = "X",aspect_ratio = :equal,legend = :topright);
scatter!(A[1,:],A[2,:],label = "generated from f1(e)");
scatter!(B[1,:],B[2,:],label = "generated from f2(e)")

Z1 = generate_from_X(dataX2, f1, z1, μ1, logσ1, A1);
Z2 = generate_from_X(dataX1, f2, z2, μ2, logσ2, A2);

scatter(X[1,:],X[2,:],label = L"\mathrm{data \;} X",aspect_ratio = :equal,legend = :topright);
scatter!(Z1[1,:],Z1[2,:],label = L"\mathrm{reconstructed \; from \;} f_1");
scatter!(Z2[1,:],Z2[2,:],label = L"\mathrm{reconstructed \; from \;} f_2")