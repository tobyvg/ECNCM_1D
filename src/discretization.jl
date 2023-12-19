using LinearAlgebra
using SparseArrays
using Random
using Distributions
include("methods.jl")




function generate_domain_and_filters(b,I,N;spectral = false)
    J = round(Int,floor(N/I))

    a = 0

    X = collect(LinRange(a,b,I+1)[1:end-1])
    dX = X[2]-X[1]
    X = X .+ 1/2*dX

    Omega = dX*ones(size(X)[1])
    Omega = Diagonal(Omega)

    n = (I*J)

    x = collect(LinRange(a,b,n+1)[1:end-1])
    dx = x[2] - x[1]
    x = x .+1/2*dx
    omega = Diagonal(dx*ones(size(x)[1]))

    ref_x = collect(LinRange(a,b,N+1)[1:end-1])
    ref_dx = ref_x[2] - ref_x[1]
    ref_x = ref_x .+1/2*ref_dx
    ref_omega = Diagonal(ref_dx*ones(size(ref_x)[1]))

    interpolation_matrix = construct_weighted_interpolation_matrix(ref_x,x)



    mapper = dx*ones((J,size(X)[1]))

    inner_product(a,b,omega) = sum(a .* (omega*b),dims = 1)'

    IP(a,b,omega = Omega) = inner_product(a,b,omega)
    ip(a,b,omega = omega) = inner_product(a,b,omega)
    ref_ip(a,b,omega = ref_omega) = inner_product(a,b,omega)

    INTEG(a) = IP(a,ones(size(a)))
    integ(a) = ip(a,ones(size(a)))
    ref_integ(a) = ref_ip(a,ones(size(a)))
    if spectral
        W, R = spectral_filter(X,x,domain_range)
    else
        W, R = gen_W(mapper),gen_R(mapper)
    end
    domain_descriptors = b,interpolation_matrix,(N,I,J),(X,x,ref_x),(Omega,omega,ref_omega),(W,R),(IP,ip,ref_ip),(INTEG,integ,ref_integ)
    return domain_descriptors
end

function fourier_basis(x,domain_range)
    N = size(x)[1]
    basis = Array{Float64}(undef,  N,0)
    for k in 1:floor(Int,N/2)
        basis_cos = cos.((k-1)*2*pi * x/domain_range)
        basis = [basis;;basis_cos/sqrt(basis_cos'*basis_cos)]
        basis_sin = sin.(k*2*pi * x/domain_range)
        basis = [basis;;basis_sin/sqrt(basis_sin'*basis_sin)]
    end
    if N % 2 == 1
        basis_cos = cos.((floor(Int,N/2) )*2*pi*x/domain_range)
        basis = [basis;;basis_cos /sqrt(basis_cos'*basis_cos)]
    end
    return basis
end

function spectral_filter(X,x,domain_range)
    basis = fourier_basis(x,domain_range)
    basis_bar = fourier_basis(X,domain_range)
    spectral_filter = spzeros(size(X)[1],size(x)[1])

    for i in 1:size(X)[1]
        spectral_filter[i,i] = 1
    end

    ratio = sqrt(size(X)[1]/size(x)[1])
    W = ratio*basis_bar*spectral_filter*basis'
    R = 1/ratio*basis*spectral_filter'*basis_bar'
    return W,R
end


function gen_W(mapper)
    I = size(mapper)[1]
    J = size(mapper)[2]
    mat = spzeros((J,I*J))
    for j in 1:J
        surface = sum(mapper[:,j])
        for i in 1:I
            mat[j,(j-1)*I + i] = (1/surface)*mapper[i,j]
        end
    end
    return mat
end

function gen_R(mapper)
    I = size(mapper)[1]
    J = size(mapper)[2]
    mat = spzeros((I*J,J))
    for j in 1:J
        for i in 1:I
            mat[(j-1)*I+i,j] = 1
        end
    end
    return mat
end










function RK4(u,x,t,dt,f)
    k1 = f(u,x,t)
    k2 = f(u.+dt*k1/2,x,t+dt/2)
    k3 = f(u.+dt*k2/2,x,t+dt/2)
    k4 = f(u .+ dt*k3,x,t+dt)
    return 1/6*(k1.+2*k2.+2*k3.+k4)
end



function gen_fourier(domain_range,min_fourier_mode,max_fourier_mode;scaling = 1,offset = 0,a = NaN,b = NaN)
    frequency = (2*pi)/domain_range
    x = collect(LinRange(0,2*pi/frequency,1000))
    if min_fourier_mode == 0 && max_fourier_mode == 0
        max_mode = 1
        coeffs = zeros(2*max_mode)
    else
        max_mode = rand(min_fourier_mode:max_fourier_mode)
        coeffs = sign.(rand(Uniform(-1.,1.),2*max_mode)).*rand(Uniform(0.5,1),2*max_mode)
        #coeffs = sign.(rand(Normal(0,0.5),2*max_mode)).*rand(Normal(0.5,0.5),2*max_mode)
    end
    function funct(x,coeffs = coeffs,frequency = frequency,offset = offset,max_mode = max_mode,scaling = scaling)
        eval = scaling/sqrt(max_mode)*sum([coeffs[i]*sin.(i*frequency*x) .+ coeffs[i+max_mode]*cos.(i*frequency*x) for i in 1:max_mode],dims =1)[1]
        eval = eval .+ offset
        return eval
    end

    val  = funct(0)
    function BC_correction_funct(x,a = a,b= b,frequency = frequency,val = val)
        if isnan(a) == false && isnan(b) == false
            return - val .+ a .+ (frequency/(2*pi))*(b  - a ) .*x
        elseif isnan(a) == false
            return - val .+ a
        elseif isnan(b) == false
            return - val .+ b
        else
            return 0
        end
    end
    full_funct(x) = funct(x) .+ BC_correction_funct(x)
    return full_funct
end




function process_HR_solution(us,dus,ts,domain_descriptors,f,return_primes = false)

    domain_range,interpolation_matrix,(N,I,J),(X,x,ref_x),(Omega,omega),(W,R),(IP,ip,ref_ip),(INTEG,integ,ref_integ) = domain_descriptors
    Es =  (1/2)*ref_ip(us,us)
    dEs = ref_ip(us,dus)


    us = interpolation_matrix*us
    dus = interpolation_matrix*dus


    us_bar = W*us
    phys_dus = f(us_bar,X,ts)
    dus_bar = W*dus

    Es_bar = (1/2)*IP(us_bar,us_bar)
    Es_prime = Es .- Es_bar



    dEs_bar = IP(us_bar,dus_bar)
    dEs_prime = dEs .- dEs_bar

    if return_primes
        HR_us_bar = R*us_bar
        HR_us_prime = us .- HR_us_bar

        HR_dus_bar = R*dus_bar
        HR_dus_prime = dus .- HR_dus_bar

    else
        us = []
        dus = []
        HR_us_bar = []
        HR_us_prime = []

        HR_dus_bar = []
        HR_dus_prime = []


    end


    return Dict([("u",us),("du",dus),("phys_du",phys_dus),("t",ts),("u_bar", us_bar), ("du_bar", dus_bar),("E",Es),("E_bar",Es_bar),("E_prime",Es_prime),("HR_u_bar",HR_us_bar),("u_prime",HR_us_prime),("HR_du_bar",HR_dus_bar),("du_prime",HR_dus_prime),("dE",dEs),("dE_bar",dEs_bar),("dE_prime",dEs_prime)])
end







function pre_alloc_simulation(u0,x,dt,T,f,F,save_every)

    f_plus_forcing(u,x,t) = f(u,x,t) .+ F


    t= 0.
    u = u0

    ############
    storage_size = floor(Int,(T/dt)/save_every) + 1
    us = zeros(size(u)[1],storage_size )
    dus = zeros(size(u)[1],storage_size )
    ts = zeros(storage_size)
    ############

    save_counter = save_every + 1
    counter = 1
    while round(t,digits =10) <= T
        du = RK4(u,x,t,dt,f_plus_forcing)
        if save_counter > save_every && counter <= storage_size
            #############################################

            us[:,counter] = u
            dus[:,counter] = f(u,x,t) #-forcing
            ts[counter] = t

            #############################################
            save_counter = 1
            counter += 1
        end
        u = u .+ dt*du
        t += dt
        t = round(t,digits = 10)
        save_counter += 1
    end
    return us,dus,ts'
end

function simulation(u0,x,dt,T,f;F = 0,save_every= 1,pre_allocate = false)

    if pre_allocate
        @assert length(size(u0))[1] == 1 "Pre-allocation only supported for initial condition of size [any,1]"
        us,dus,ts = pre_alloc_simulation(u0,x,dt,T,f,F,save_every)
    else
        f_plus_forcing(u,x,t) = f(u,x,t) .+ F

        t= 0.
        u = u0

        ############
        us = Array{Float64}(undef, size(u0)[1], 0)
        dus = Array{Float64}(undef,  size(u0)[1], 0)
        ts = Array{Float64}(undef,  1,0)
        ############
        save_counter = save_every + 1
        counter = 1
        round_t = t
        while round_t <= T
            du = RK4(u,x,t,dt,f_plus_forcing)
            if save_counter > save_every
                #############################################
                us = [us u]

                dus = [dus f(u,x,t)] #-forcing
                if length(size(u0)) > 1
                    ts= [ts t*ones(1,size(u0)[2])]
                else
                    ts = [ts t]
                end
                #############################################
                save_counter = 1
                counter += 1

            end
            u = u .+ dt*du

            t += dt
            save_counter += 1
            round_t = stop_gradient() do
                round(t,digits =10)
            end
        end
    end
    return us,dus,ts

end

function gen_conv_stencil(weights;stride = 1)
    sizes = [size(weights)[1]]
    conv_stencil = conv_NN(sizes,[1,1],[stride],false)

    conv_stencil[1].weight[:,:,1] = reverse(weights)
    return conv_stencil
end

function gen_conv_burgers_dudt(viscosity)
    conv_D1 = gen_conv_stencil([-1.,0.,1.])
    conv_D2 = gen_conv_stencil([1.,-2.,1.])
    function burgers_dudt(u,x,t,D1=conv_D1,D2 = conv_D2,viscosity =  viscosity)
        if length(size(u)) == 1
            u = reshape(u,(size(u)[1],1))
        end

        dx = x[2]-x[1]
        u = reshape(u,(size(u)[1],1,size(u)[2]))

        A = (1/3)*((1/(2*dx))*(D1(u.^2) .+ u[2:end-1,:,:] .* D1(u)))
        B = (1/(dx^2))*D2(u)

        return reshape(-A .+ viscosity*B,(size(u)[1]-2,size(u)[3]))
    end
    return burgers_dudt
end


function gen_conv_KdV_dudt()
    conv_D1 = gen_conv_stencil([-1.,0.,1.])
    conv_D3 = gen_conv_stencil([-1.,2.,0.,-2.,1.])

    function KdV_dudt(u,x,t,D1=conv_D1,D3 = conv_D3)
        if length(size(u)) == 1
            u = reshape(u,(size(u)[1],1))
        end

        dx = x[2]-x[1]
        u = reshape(u,(size(u)[1],1,size(u)[2]))

        A = (1/3)*((1/(2*dx))*(D1(u.^2) .+ u[2:end-1,:,:] .* D1(u)))
        B = 1/(2*dx^3)*D3(u)

        return reshape(-(6*A[2:end-1,:,:] .+ B),(size(u)[1]-4,size(u)[3]))
        #return reshape((1/(2*dx))*D1(u),(size(u)[1]-2,size(u)[3]))
    end
    return KdV_dudt
end

function padding_wrapper(f;pad_size=NaN,eval_BCs = false,as = 0 ,bs = 0)
    if isnan(pad_size)
        pad_size = Int((20-size(f(ones(20),ones(20),0))[1])/2)
    end

    function pad_eval_f(u,x,t,f = f;pad_size = pad_size,as = as,bs = bs,eval_BCs = eval_BCs)
        if eval_BCs
            if length(size(t)) == 0
                t = [t]'
            end

            if length(size(u)) == 2
                t = stop_gradient() do
                t[1,1]*ones((1,size(u)[2]))
                end
            end


            if as != 0
                eval_a = as(t)
            end
            if bs != 0
                eval_b = bs(t)
            end
            pad_u = padding(u,pad_size,eval_a,eval_b)

        else

            pad_u = padding(u,pad_size,as,bs)

        end
        return f(pad_u,x,t)
    end
    return pad_eval_f
end

function gen_rand_condition(domain_descriptors,f,offset,min_mode,max_mode;scaling = 1,in_outflow = false)

    domain_range,interpolation_matrix,(N,I,J),(X,x,ref_x),(Omega,omega),(W,R),(IP,ip,ref_ip),(INTEG,integ,ref_integ) = domain_descriptors

    b(t) = NaN*t

    if in_outflow

        a = gen_fourier(2*pi,min_mode,max_mode,offset = offset,scaling = scaling)

        F = 0.5*gen_fourier(domain_range,min_mode,max_mode,offset = 0,scaling = scaling)(ref_x)

        u0 = gen_fourier(domain_range,0,0,offset= offset,a = a(0),b = b(0),scaling = scaling)(ref_x)
    else
        a(t) = NaN*t
        F = 0*ref_x
        u0 =  gen_fourier(domain_range,min_mode,max_mode,offset = offset,a =a(0),b = b(0),scaling = scaling)(ref_x)
    end



    BC_f = padding_wrapper(f,eval_BCs = true,as =a,bs = b)


    return u0,BC_f,a,b,F
end
