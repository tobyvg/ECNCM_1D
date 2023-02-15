using LinearAlgebra
using Flux
include("methods.jl")

using Random
using Zygote
using Distributions

stop_gradient(f) = f()
Zygote.@nograd stop_gradient


function enforce_momentum_conservation(stencil,symm = "none",mom_cons = true)
    stencil_neighbors = round(Int,(size(stencil)[1]-1)/2)
    if symm == "skew-symm"
        bar = [[stencil[i] for i in 1:stencil_neighbors];[0];reverse([-stencil[i] for i in 1:stencil_neighbors])]
    elseif symm == "symm"
        bar = [[stencil[i] for i in 1:(stencil_neighbors+1)];reverse([stencil[i] for i in 1:stencil_neighbors])]
        if mom_cons
            bar = bar .- mean(bar)
        end
    else
        if mom_cons
            bar = stencil .- mean(stencil)
        else
            bar = stencil
        end
    end
    return bar
end





function gen_select_mat(x,neighbors,reduction_fold = 1)
    J = size(x)[1]
    K = reduction_fold*(2*neighbors + 1)
    mat = zeros(Int,(J,K))
    for j in 1:J
        for k in -neighbors:neighbors
            for l in 1:reduction_fold
                mat[j,reduction_fold*(k + neighbors) + 1 + (l-1)] = index_converter(reduction_fold*(j+k) +(l-reduction_fold),reduction_fold*J)
            end
        end
    end
    return mat'
end

function sp_mat_mul(quantity,stencil)
    stencil_width = Int((size(stencil)[1] - 1)/2)
    select_mat = stop_gradient() do
        gen_select_mat(quantity,stencil_width)[:,stencil_width + 1: end - stencil_width]
    end
    return vcat([stencil' * quantity[select_mat[:,i],:] for i in 1:size(select_mat)[2]]...)
end







function init_stencil(stencil_width,coeff = 0.4)
    stencil = exp.(-coeff*collect(-stencil_width:stencil_width).^2).*(-ones(2*stencil_width + 1)).^collect(1:2*stencil_width + 1)
    return stencil .* sign.(rand(Uniform(-1,1)))
end


function new_diss_form(stencil_i,p_i,p_j,stencil_j,u,diagonals)
    mult = sp_mat_mul(u,stencil_j)

    res = stop_gradient() do
        zeros(size(p_j[1]))
    end

    int_res = stop_gradient() do
       zeros(size(p_j[1]))
    end
    for i in 0:diagonals
        int_res = int_res .+ p_j[i+1] .* circshift(mult,i - diagonals)
    end
    for i in 0:diagonals
        res = res .+ circshift(reverse(p_i)[1+i],i) .* circshift(int_res,i)
    end

    return -sp_mat_mul(res[diagonals + 1:end-diagonals,:],reverse(stencil_i))
end

function constr_layer(stencil1,diagonal_vec,stencil2,u,diagonals,diss = false)

    mult = sp_mat_mul(u,stencil2)

    res = stop_gradient() do
        zeros(size(diagonal_vec[1]))
    end
    if diss
        int_res = stop_gradient() do
           zeros(size(diagonal_vec[1]))
        end
        for i in 0:diagonals
            int_res = int_res .+ diagonal_vec[i+1] .* circshift(mult,i - diagonals)
        end
        for i in 0:diagonals
            res = res .+ circshift(reverse(diagonal_vec)[1+i+diagonals],i) .* circshift(int_res,i)
        end
    else
        for i in 0:diagonals
            res = res .+ diagonal_vec[i+1] .* circshift(mult,i - diagonals)
        end
        for i in 1:diagonals
            res = res .+ circshift(diagonal_vec[1+i+diagonals],i) .* circshift(mult,i)
        end
    end


    return sp_mat_mul(res[diagonals + 1:end-diagonals,:],stencil1)
end

function skew_symm_form(stencil1,Phi,stencil2,u,diagonals)


    cons_1 = constr_layer(stencil1,Phi,stencil2,u,diagonals)
    cons_2 = constr_layer(reverse(stencil2),reverse(Phi),reverse(stencil1),u,diagonals)
    cons = cons_1 .- cons_2
    return cons
end



function exchange_form(stencil1,Phi,stencil2,u,diagonals)
    ex = constr_layer(stencil1,Phi,stencil2,u,diagonals)
    return ex
end

function gen_model(f,constraints,supply_s,dissipation,NN_descriptors;stencils = 0)

    if constraints
        @assert constraints == supply_s  "If you use constraints you need to supply the SGS variables"
    end


    (stencil_width,kernel_widths,channels,diagonals) = NN_descriptors

    @assert size(kernel_widths)[1] == size(channels)[1] + 1 "Supply n kernel_widths and n-1 channels"

    if stencils != 0
        stencil_width = Int((size(stencils[1])[1]-1)/2)
    end

    physics_width = Int((20-size(f(ones(20),ones(20),0))[1])/2)
    kernel_widths = [kernel_widths;]

    if constraints
        if dissipation
            #channels = [[3]; channels; [2*diagonals*3 + 2*diagonals + 5]]
            channels = [[3]; channels; [2*diagonals*3 + 4*diagonals + 7]]
        else
            channels = [[3]; channels; [2*diagonals*3  + 3]]
        end
    else
        channels = [[3]; channels; [2]]
    end

    conv = conv_NN(2*kernel_widths .+ 1,channels,0)

    conv_pad_size = sum(kernel_widths)

    if stencils == 0


        M11 = init_stencil(stencil_width)
        M12 = init_stencil(stencil_width)

        M21 = init_stencil(stencil_width)
        M22 = init_stencil(stencil_width)

        M31 = init_stencil(stencil_width)
        M32 = init_stencil(stencil_width)


        M1 = init_stencil(stencil_width)
        M2 = init_stencil(stencil_width)

        M3 = init_stencil(stencil_width)
        M4 = init_stencil(stencil_width)

        stencils = (M11,M12,M21,M22,M31,M32,M1,M2,M3,M4)
    end




    BC_f = padding_wrapper(f)



    model(u,s,x,as = 0,bs = 0,stencils = stencils,BC_f = BC_f,conv = conv,constraints = constraints,dissipation = dissipation,pad_sizes = (physics_width,stencil_width,conv_pad_size),supply_s = supply_s,diagonals = diagonals;channel = "all") = skew_symm_NN(u,s,x,as,bs,stencils,BC_f,conv,constraints,dissipation,pad_sizes,supply_s,diagonals;channel)

    return model, (conv,stencils)

end


function skew_symm_NN(u,s,x,as,bs,stencils,BC_f,conv,constraints,dissipation,pad_sizes,supply_s,diagonals;channel)

    #domain_range,(I,J),(X,x),(Omega,omega),(W,R),(IP,ip),(INTEG,integ) = domain_descriptors


    (physics_width,stencil_width,conv_pad_size) = pad_sizes

    if constraints
        conv_pad_size = conv_pad_size + stencil_width + diagonals
    else
        conv_pad_size += 1
    end





    pad2 = BC_f(u,x,0,as = as,bs = bs,eval_BCs = false,pad_size = conv_pad_size + physics_width)

    du = pad2[conv_pad_size + 1:end - conv_pad_size,:]
    pad1 = padding(u,conv_pad_size,as,bs)
    #pad2 = padding(du,conv_pad_size)
    pad3 = padding(s,conv_pad_size,as,bs,anti_symm_outflow = true)



    if supply_s
        pads = [pad1;pad2;pad3]
    else
        pads = [pad1;pad2; 0 .* pad3]
    end

    pads = reshape(pads,(size(pad1)[1],3,size(pad1)[2]))


    p = conv(pads)


    if constraints
        (M11,M12,M21,M22,M31,M32,M1,M2,M3,M4) = stencils


        M11_bar = enforce_momentum_conservation(M11)
        M12_bar = enforce_momentum_conservation(M12)
        M31_bar = enforce_momentum_conservation(M31)
        M1_bar = enforce_momentum_conservation(M1)
        M3_bar = enforce_momentum_conservation(M3)


        pad_u = padding(u,2*stencil_width + diagonals,as,bs)
        pad_s = padding(s,2*stencil_width + diagonals,0*as,0*bs,anti_symm_outflow = true)


        add = 1+2*diagonals

        select_vec = collect(1:add)

        Phi1 = Tuple(p[:,i,:] for i in select_vec)
        Phi2 = Tuple(p[:,i,:] for i in select_vec .+ add)
        Phi3 =  Tuple(p[:,i,:] for i in select_vec .+ 2*add)

        cons_u = skew_symm_form(M11_bar,Phi1,M12_bar,pad_u,diagonals)

        cons_s = skew_symm_form(M21,Phi2,M22,pad_s,diagonals)


        ex_u = exchange_form(M31_bar,Phi3,M32,pad_s,diagonals)
        ex_s = -exchange_form(reverse(M32),reverse(Phi3),reverse(M31_bar),pad_u,diagonals)


        pu = cons_u  .+ ex_u
        ps = cons_s .+ ex_s



        if dissipation

            Psi1 = Tuple(p[:,i,:] for i in select_vec[1:end-diagonals] .+ 3*add)

            Psi2 = Tuple(p[:,i,:] for i in select_vec[1:end-diagonals] .+ 4*add .- 1*diagonals)

            Psi3 = Tuple(p[:,i,:] for i in select_vec[1:end-diagonals] .+ 5*add .- 2*diagonals)
            Psi4 = Tuple(p[:,i,:] for i in select_vec[1:end-diagonals] .+ 6*add .- 3*diagonals)

            #diss_u = diss_form(M41_bar,p1,pad_u,diagonals)
            #diss_s = diss_form(M51,p2,pad_s,diagonals)

            diss_u = new_diss_form(M1_bar,Psi1,Psi1,M1_bar,pad_u,diagonals) .+ new_diss_form(M3_bar,Psi3,Psi3,M3_bar,pad_u,diagonals) .+ new_diss_form(M1_bar,Psi1,Psi2,M2,pad_s,diagonals) .+ new_diss_form(M3_bar,Psi3,Psi4,M4,pad_s,diagonals)
            diss_s = new_diss_form(M2,Psi2,Psi1,M1_bar,pad_u,diagonals)  .+ new_diss_form(M4,Psi4,Psi3,M3_bar,pad_u,diagonals) .+  new_diss_form(M2,Psi2,Psi2,M2,pad_s,diagonals) .+ new_diss_form(M4,Psi4,Psi4,M4,pad_s,diagonals)

            pu = pu .+ diss_u
            ps = ps .+ diss_s
        else
            diss_u = 0*u
            diss_s = 0*u
        end

    else
        pu = 1/(x[2]-x[1]).*sp_mat_mul(p[:,1,:],[0,-1,1])
        if supply_s
            ps = p[:,2,:][2:end-1,:]
        else
            ps = 0*pu
        end
    end



    pu = pu .+ du
    if channel == "all"
        return pu,ps
    elseif channel == "conservative"
        return cons_u,cons_s
    elseif channel == "dissipative"
        return diss_u,diss_s
    elseif channel == "exchange"
        return ex_u,ex_s
    elseif channel == "physics"
        return du,0*du
    elseif channel == "closure"
        return pu .- du,ps
    else
        return 0*pu,0*ps
    end
end




function loss_function(us,dus,ss,dss,as,bs,domain_descriptors,model;subgrid_loss = true)

    domain_range,interpolation_matrix,(N,I,J),(X,x,ref_x),(Omega,omega,ref_omega),(W,R),(IP,ip,ref_ip),(INTEG,integ,ref_integ) = domain_descriptors

    #us,dus,ss,dss,as,bs = stop_gradient() do
    #    [hcat([k[:,i,j] for i in 1:size(k)[2] for j in 1:size(k)[3]]...)  for k in (us,dus,ss,dss,as,bs)]
    #end

    pred_dus,pred_dss = model(us,ss,X,as,bs)



    l1 = Flux.Losses.mse(dus, pred_dus)
    l2 = Flux.Losses.mse(dss, pred_dss )
    if subgrid_loss
        return l1  + l2
    else
        return l1
    end
end


function interpolate_unsteady_BCs(t,cond,traj_dt)
    interval = round(Int,t[1]/traj_dt)+1
    w1 = t[1]/traj_dt+1 - interval
    w2 = 1 - w1
    if interval < size(cond)[2]
        return w1*cond[:,interval,:] + w2*cond[:,interval+1,:]
    else
        return cond[:,end,:]
    end
end

function trajectory_loss_function(us,ss,as,bs,Fs,domain_descriptors,model,traj_dt,T;subgrid_loss = true)


    domain_range,interpolation_matrix,(N,I,J),(X,x,ref_x),(Omega,omega,ref_omega),(W,R),(IP,ip,ref_ip),(INTEG,integ,ref_integ) = domain_descriptors

    traj_us,traj_ss = stop_gradient() do
        [hcat([k[:,i,j] for i in 1:size(k)[2] for j in 1:size(k)[3]]...)  for k in (us,ss)]
    end



    BC_as(t,as = as,traj_dt = traj_dt) = interpolate_unsteady_BCs(t,as,traj_dt)
    BC_bs(t,bs = bs,traj_dt = traj_dt) = interpolate_unsteady_BCs(t,bs,traj_dt)

    current_f = model_wrapper(model,as = BC_as,bs = BC_bs,eval_BCs= true)

    init_cond = [us[:,1,:];ss[:,1,:]]
    NN_var,NN_dus,NN_ts = simulation(init_cond ,X,traj_dt,T,current_f,F = Fs[:,1,:],save_every = 1,pre_allocate = false)


    NN_us = NN_var[1:size(X)[1],:]
    NN_ss = NN_var[size(X)[1]+1:2*size(X)[1],:]

    l1 = Flux.Losses.mse(NN_us[:,size(us)[3]+1:end],traj_us[:,size(us)[3]+1:end])
    l2 = Flux.Losses.mse(NN_ss[:,size(us)[3]+1:end],traj_ss[:,size(us)[3]+1:end])


    if subgrid_loss
        return l1  + l2
    else
        return l1
    end #+ 1/3*emb_l3# + emb_l2 + emb_l3 #l1 + l3 + l4

end




function model_wrapper(model;as = 0,bs = 0,eval_BCs = false)
    function BC_model(u,x,t;model = model,as = as,bs = bs,eval_BCs = eval_BCs)
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
            else
                eval_a = as
            end
            if bs != 0
                eval_b = bs(t)
            else
                eval_b =bs
            end


            pred = model(u[1:size(x)[1],:],u[size(x)[1]+1:2*size(x)[1],:],x,eval_a,eval_b)
        else
            pred = model(u[1:size(x)[1],:],u[size(x)[1]+1:2*size(x)[1],:],x,as,bs)
        end

        return [pred[1];pred[2]]
    end
    return BC_model
end









function padding(vec,pad_size,a = 0, b = 0;anti_symm_outflow = false)

    if a == 0 || b == 0

        a,b = stop_gradient() do
            NaN * ones(size(vec)[2]),NaN * ones(size(vec)[2])
        end
    end
    if length(size(vec)) == 1
        vec = reshape(vec,(size(vec)[1],1))
    end

    l = size(vec)[1]

    @assert pad_size <= l "Padding larger than vector, structure not preserved => lower your cumulative\nfilter widths (stencil + convolutional + diagonal) from " * string(pad_size) * " to less than " * string(length)


    front = stop_gradient() do
        Array{Float64}(undef, pad_size,0)
    end
    back = stop_gradient() do
        Array{Float64}(undef, pad_size,0)
    end

    for i in 1:size(vec)[2]
        if isnan(a[i]) && isnan(b[i])
            front = [ front vec[end-pad_size+1:end,i]]
            back = [ back vec[1:pad_size,i]]
        elseif isnan(a[i]) && isnan(b[i]) == false
            if anti_symm_outflow
                back = [ back reverse(vec[end-pad_size+1:end,i])]
                front = [ front -reverse(vec[1:pad_size,i])]
            else
                back = [ back (2*vec[end,i] .- 2*(vec[end,i]-b[i]) .- reverse(vec[end-pad_size+1:end,i]))]
                front = [ front reverse(vec[1:pad_size,i])]
            end
        elseif isnan(b[i]) && isnan(a[i]) == false
            #front = [ front a[i]*vec[end-pad_size+1:end,i].^0]
            if anti_symm_outflow
                back = [ back -reverse(vec[end-pad_size+1:end,i])]
                front = [ front reverse(vec[1:pad_size,i])]
            else
                back = [ back reverse(vec[end-pad_size+1:end,i])]
                front = [ front ((2*vec[1,i]) .-2*(vec[1,i]-a[i]) .-reverse(vec[1:pad_size,i]))]
            end
        else
            #front = [ front a[i]*vec[end-pad_size+1:end,i].^0]
            #back = [ back b[i]*vec[1:pad_size,i].^0]
            if anti_symm_outflow
                back = [ back reverse(vec[end-pad_size+1:end,i])]
                front = [ front reverse(vec[1:pad_size,i])]
            else
                back = [ back (2*vec[end,i] .- 2*(vec[end,i]-b[i]) .- reverse(vec[end-pad_size+1:end,i]))]
                front = [ front ((2*vec[1,i]) .-2*(vec[1,i]-a[i]) .-reverse(vec[1:pad_size,i]))]
            end
        end
    end
    return [front; vec ; back]
end

function subgrid_gradients(u_prime,du_prime,domain_descriptors,subgrid_filter)
    domain_range,interpolation_matrix,(N,I,J),(X,x,ref_x),(Omega,omega,ref_omega),(W,R),(IP,ip,ref_ip),(INTEG,integ,ref_integ) = domain_descriptors
    function extract_jacobian(u_prime,subgrid_filter = subgrid_filter,I=I,J=J)
        jac = jacobian(subgrid_filter,u_prime)[1]
        flat_jac = zeros(I*J)
        for i in 1:I
            for j in 1:J
                flat_jac[j + (i-1)*J] += jac[i,j + (i-1)*J]
            end
        end
        return flat_jac
    end
    full_jac = zeros(I,size(u_prime)[2])

    for i in 1:size(u_prime)[2]
        full_jac[:,i] .+= R'*(extract_jacobian(u_prime[:,i]) .* du_prime[:,i])
    end
    return full_jac
end


function gen_NN_subgrid_filter(layers,domain_descriptors,outflow = false)
     domain_range,interpolation_matrix,(N,I,J),(X,x,ref_x),(Omega,omega,ref_omega),(W,R),(IP,ip,ref_ip),(INTEG,integ,ref_integ) = domain_descriptors
     NN = NeuralNetwork(layers,bias = false, activation_function = tanh)
     select_mat = gen_subgrid_filter_select_mat(domain_descriptors)
     Px = stop_gradient() do
            reverse(Matrix{Float64}(LinearAlgebra.I, J, J),dims = 2)
     end
     function subgrid_filter(u_primes,NN = NN, select_mat = select_mat,domain_descriptors = domain_descriptors,Px = Px,outflow = outflow)
        domain_range,interpolation_matrix,(N,I,J),(X,x,ref_x),(Omega,omega,ref_omega),(W,R),(IP,ip,ref_ip),(INTEG,integ,ref_integ) = domain_descriptors

        f1 = vcat([NN(u_primes[select_mat[:,i],:]) for i in 1:size(select_mat)[2]]...)
        f2 = vcat([NN(Px*u_primes[select_mat[:,i],:]) for i in 1:size(select_mat)[2]]...)
        filtered = f1 .- f2
        return filtered
    end
    return subgrid_filter,NN
end


function NeuralNetwork(layers;bias = true,activation_function = relu)
    storage = []
    for i in 1:size(layers)[1]-1
        if i == size(layers)[1]-1
            storage = [storage; Dense(layers[i],layers[i+1],bias = bias)]
        else
            storage = [storage; Dense(layers[i],layers[i+1],activation_function,bias = bias)]
        end
    end
    return Chain((i for i in storage)...)
end

function conv_NN(sizes,channels,strides = 0,bias = true)
    if strides == 0
        strides = ones(Int,size(sizes)[1])
    end
    storage = []
    for i in 1:size(sizes)[1]
        if i == size(sizes)[1]
            storage = [storage;Conv((sizes[i],), channels[i]=>channels[i+1],stride = strides[i],pad = (0,),bias = bias)]
        else
            storage = [storage;Conv((sizes[i],), channels[i]=>channels[i+1],stride = strides[i],pad = (0,) ,relu,bias = bias)]
        end
    end
    return Chain((i for i in storage)...)
end


function gen_subgrid_filter_select_mat(domain_descriptors)
    domain_range,interpolation_matrix,(N,I,J),(X,x,ref_x),(Omega,omega,ref_omega),(W,R),(IP,ip,ref_ip),(INTEG,integ,ref_integ) = domain_descriptors
    selects = reshape(collect(1:size(x)[1]),(J,I))
    return selects
end

function gen_t_stencil(params,domain_descriptors,outflow)
    domain_range,interpolation_matrix,(N,I,J),(X,x,ref_x),(Omega,omega,ref_omega),(W,R),(IP,ip,ref_ip),(INTEG,integ,ref_integ) = domain_descriptors
    if outflow
        lambda = 0.
        t_tilde = params
        Px = stop_gradient() do
            reverse(Matrix{Float64}(LinearAlgebra.I, size(t_tilde)[1], size(t_tilde)[1]),dims = 2)
        end
        I_mat = stop_gradient() do
            Matrix{Float64}(LinearAlgebra.I,  size(t_tilde)[1], size(t_tilde)[1])
        end
        t_vec = (I_mat - Px)*t_tilde .+ 1/2*lambda
    else
        t_vec = params
    end
    return t_vec

end


function gen_subgrid_filter(domain_descriptors,outflow = false)
     domain_range,interpolation_matrix,(N,I,J),(X,x,ref_x),(Omega,omega,ref_omega),(W,R),(IP,ip,ref_ip),(INTEG,integ,ref_integ) = domain_descriptors
     params = zeros(J) .+ rand(Uniform(-10^(-20),10^(-20)),J)
     select_mat = gen_subgrid_filter_select_mat(domain_descriptors)
     function subgrid_filter(u_primes,params = params, select_mat = select_mat,domain_descriptors = domain_descriptors,outflow = outflow)
        t_vec = gen_t_stencil(params,domain_descriptors,outflow)
        filtered = vcat([t_vec' *u_primes[select_mat[:,i],:] for i in 1:size(select_mat)[2]]...)
        return filtered
    end
    return subgrid_filter,params
end
    #bar = gen_S_and_K(subgrid_filter_stencil,"just_avg")

function subgrid_filter_loss(u_primes,subgrid_filter,domain_descriptors)
    domain_range,interpolation_matrix,(N,I,J),(X,x,ref_x),(Omega,omega,ref_omega),(W,R),(IP,ip,ref_ip),(INTEG,integ,ref_integ) = domain_descriptors

    filtered = subgrid_filter(u_primes)
    filtered_squared = 1/2*(filtered).^2

    target = W*(1/2*u_primes.^2)
    l1 = Flux.Losses.mse(filtered_squared ,target)


    return  l1
end

function gen_T(subgrid_filter_stencil,domain_descriptors)
    domain_range,interpolation_matrix,(N,I,J),(X,x,ref_x),(Omega,omega,ref_omega),(W,R),(IP,ip,ref_ip),(INTEG,integ,ref_integ) = domain_descriptors
    dimensions = (I,I*J)
    mat = spzeros(dimensions)
    for i in 1:I
        mat[i,1+(i-1)*J:i*J] = subgrid_filter_stencil
    end
    return mat
end

function gen_train_test_set(d,f,domain_descriptors,fraction,train_fraction)
    domain_range,interpolation_matrix,(N,I,J),(X,x,ref_x),(Omega,omega,ref_omega),(W,R),(IP,ip,ref_ip),(INTEG,integ,ref_integ) = domain_descriptors
    us,dus,ts,as,bs,Fs = d
    ending_index = Int(floor(train_fraction*size(us)[2]))

    indexes = cut_indexes(us[:,1:ending_index],fraction,randomize = false,uniform = false)

    data = Dict()
    test_data = Dict()

    BC_f = padding_wrapper(f,eval_BCs = false,as =as[:,indexes],bs = bs[:,indexes])
    data = process_HR_solution(us[:,indexes],dus[:,indexes],ts[:,indexes],domain_descriptors,BC_f,true)
    data["a"] = as[:,indexes]
    data["b"] = bs[:,indexes]
    #data["F"] = Fs[:,indexes]

    test_indexes = cut_indexes(us[:,ending_index:end],fraction,randomize = false,uniform = false) .+ ending_index .- 1

    test_BC_f = padding_wrapper(f,eval_BCs = false,as =as[:,test_indexes],bs = bs[:,test_indexes])
    test_data = process_HR_solution(us[:,test_indexes],dus[:,test_indexes],ts[:,test_indexes],domain_descriptors,test_BC_f,true)
    test_data["a"] = as[:,test_indexes]
    test_data["b"] = bs[:,test_indexes]

    return data,test_data,indexes,test_indexes
end

function gen_trajectory_data(d,f,indexes,domain_descriptors;increase_step_size,traj_steps,T_mat)
    domain_range,interpolation_matrix,(N,I,J),(X,x,ref_x),(Omega,omega,ref_omega),(W,R),(IP,ip,ref_ip),(INTEG,integ,ref_integ) = domain_descriptors



    traj_fraction = 1/traj_steps

    us,dus,ts,as,bs,Fs = d

    traj_indexes = cut_indexes(indexes,traj_fraction,randomize = false,uniform = false)

    traj_data = Dict()
    traj_data["u_bar"] = Array{Float64}(undef, I,traj_steps+ 1,0)
    traj_data["s"] = Array{Float64}(undef, I,traj_steps+ 1,0)
    traj_data["t"] = Array{Float64}(undef, 1,traj_steps+ 1,0)
    traj_data["a"] = Array{Float64}(undef, 1,traj_steps+ 1,0)
    traj_data["b"] = Array{Float64}(undef, 1,traj_steps+ 1,0)

    select_vec = collect(0:increase_step_size:increase_step_size*traj_steps)

    trajs = Array{Int64}(undef, 0)
    for i in traj_indexes
        traj = select_vec .+ i
        traj_t = ts[traj]
        if traj_t[1] < traj_t[end]
            trajs = [trajs ; traj]
        end
    end

    data_size = Int(size(trajs)[1]/(traj_steps+1))

    BC_f = padding_wrapper(f,eval_BCs = false,as =as[:,trajs],bs = bs[:,trajs])
    traj_data = process_HR_solution(us[:,trajs],dus[:,trajs],ts[:,trajs],domain_descriptors,BC_f,true)

    trajectory_data = Dict()
    trajectory_data["u_bar"] = reshape(traj_data["u_bar"],(I,traj_steps+1,data_size))
    trajectory_data["s"] = reshape(T_mat*traj_data["u_prime"],(I,traj_steps+1,data_size))
    trajectory_data["t"] = reshape(traj_data["t"],(1,traj_steps+1,data_size))
    trajectory_data["a"] = reshape(as[:,trajs],(1,traj_steps+1,data_size))
    trajectory_data["b"] = reshape(bs[:,trajs],(1,traj_steps+1,data_size))

    #### Process F ########
    F_bar  = W*interpolation_matrix*Fs[:,trajs]
    F_prime = T_mat*(interpolation_matrix*Fs[:,trajs] .-R*F_bar)

    NN_F = [F_bar ; F_prime]

    #######################
    #######################

    trajectory_data["F"] = reshape(NN_F,(2*I,traj_steps+1,data_size))
    return trajectory_data
end

function save_model(path,t_stencil,constraints,supply_s,dissipation,NN_descriptors,stencils,conv)
    path = process_path(path)
    mkpath(path)
    save(path * "T.jld","t_stencil",t_stencil)
    save(path * "model.jld","constraints",constraints,"supply_s",supply_s,"dissipation",dissipation,"NN_descriptors",NN_descriptors,"stencils",stencils,"weight_bias",[(i.weight,i.bias) for i in conv])
    print("Model saved at [" * path *"]")
end

function import_model(path,f)
    path = process_path(path)
    constraints,supply_s,dissipation,NN_descriptors,stencils,weight_bias = (load(path * "model.jld")[i] for i in ("constraints","supply_s","dissipation","NN_descriptors","stencils","weight_bias"))
    model, (conv,stencils) = gen_model(f, constraints,supply_s,dissipation,NN_descriptors,stencils = stencils)

    for i in 1:length(conv)
        conv[i].weight[:,:,:] = weight_bias[i][1]
        conv[i].bias[:] = weight_bias[i][2]
    end

    t_stencil = load(path * "T.jld")["t_stencil"]

    return model, (conv,stencils), t_stencil
end
