function rand_mixture(m1,m2,p)
    a = rand(Multinomial(1,p))[1] 
    if a == 0
        return rand(m1)
    else
        return rand(m2)
    end
end

function new_sample(nz,f,m1,m2,p)
    a = rand_mixture(m1,m2,p)
    return f(a)
end

function new_data(n,nz,f,m1,m2,p)
    new = hcat([new_sample(nz,f,m1,m2,p) for i in 1:n]...)
    return new
end

function encoding(x)
    HID = A(x)
    zs = z(μ(HID),logσ(HID))
end

function data_encoding(dataT)
    return hcat([encoding(dataT[i]) for i in 1:ndat]...)
end