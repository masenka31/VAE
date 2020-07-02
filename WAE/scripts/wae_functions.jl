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
