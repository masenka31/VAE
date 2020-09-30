"""
All defined functions for counting labels,
generating new data and training model.
"""

# TRAINING FUNCTIONS
# count labels for specific data point
# NaN values correction 
function lik(x)
    HID = [A[i](x) for i in 1:comp]
    zs = [z(μ[i](HID[i]), logs[i](HID[i])) for i in 1:comp]
    expo = map(1:comp) do i
        temp = f[i](zs[i])
        vc = -sum((x .- temp).^2)
    end
    lbl = softmax(-expo .* α[:])
    return lbl
end

# lik function without NaN values correction
function _lik(x)
    HID = [A[i](x) for i in 1:comp]
    zs = [z(μ[i](HID[i]), logs[i](HID[i])) for i in 1:comp]
    expo = map(1:comp) do i
        temp = f[i](zs[i])
        vc = -sum((x .- temp).^2)
    end
    max_exp = maximum(expo)
    lbl = [exp(expo[i] - max_exp) * α[i] for i in 1:comp] 
    soucet = sum(lbl)
    lbl = lbl ./ soucet
    return lbl
end

# count labels for the whole dataset
function lik_n(dataT)
    n = size(dataT,1)
    LI = zeros(n, comp)
    for i in 1:n
        li = lik(dataT[i])
        for j in 1:comp
            LI[i,j] = li[j]
        end
    end
    return LI
end

# recontstruction loss for specific component i
function rec(x, i)
    HID = A[i](x)
    zsample = z(μ[i](HID), logs[i](HID))
    return 0.5 * b * sum((x .- f[i](zsample)).^2)
end

# GENERATE DATA
# generate new data from decoder on N(0,I)
# generate the same amount of samples from each decoder 
function generate_new(f,n)
    new = [0;0]
    for i in 1:comp
        latent = rand(MvNormal(zeros(nz),I),n)
        temp = hcat([f[i](latent[:,j]) for j in 1:n]...)
        new = hcat(new,temp)
    end
    return new[:,2:end]
end

# generate new data taking α as a parameter
# to create representative data
function generate_new(f,n,α)
    new = zeros(nx)
    comp_index = collect(1:size(f,1))
    for i in 1:n
        k = rand(Multinomial(1,α[:]))
        ind = dot(k,comp_index)
        latent = rand(MvNormal(zeros(nz),I))
        temp = f[ind](latent)
        new = hcat(new,temp)
    end
    return Float64.(new[:,2:end])
end

# create labels in a format 1 to comp
# for better visualization
function create_labels(LI)
    lb = round.(LI)
    ind = collect(1:size(LI,2))
    labels = lb*ind
end

# create labels for newly generated data
function create_labels_new(data)
    dataT = [data[:,i] for i in 1:size(data,2)]
    LI = lik_n(dataT)
    labels = create_labels(LI)
end