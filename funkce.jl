# Zde jsou všechny plošně používané funkce.
# - funkce na generaci dat
#    - banana
#    - několik gaussovek v kruhu
# - funkce na generování z dekodéru
#    - v 1d
#    - ve více dimenzích + decoder"


# n je počet vygenerovaných dat, f je dekódovací funkce
function generate_data(n,f)
    data = zeros(n)
    for i in 1:n
        test_z = rand(Normal(0,1))
        data[i] = f([test_z])[1]
    end
    return data
end

# nz je dimenze latentních proměnných, f dekódovací funkce
function decoder(nz,f)
    test_x = rand(MvNormal(zeros(nz),I))
    return f(test_x)
end

# n je počet generovaných bodů, dim je výstupní dimenze
function generate_data(n,nz,dim,f)
    new = zeros(n,dim)
    for i in 1:n
        a = decoder(nz,f)
        for j in 1:dim
            new[i,j] = a[j]
        end
    end
    return new
end

# N je počet gaussovek, n je počet bodů v každé gaussovce
# r je poloměr kruhu, s volitelný rozptyl gaussovek
function generate_circle(N,n,r,s)
    data = [0;0]
    for i in 1:N
        uhel = 2*i*pi/N
        mu = [r*cos(uhel); r*sin(uhel)]
        x = rand(MvNormal(mu,s*I),n)
        data = hcat(data,x)
    end
    return data[:,2:end]
end