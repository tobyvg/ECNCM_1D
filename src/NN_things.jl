using LinearAlgebra
using Flux
include("methods.jl")

using Random
using Zygote
using Distributions

stop_gradient(f) = f()
Zygote.@nograd stop_gradient










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
    end

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









function padding(vec,pad_size,a = 0, b = 0;anti_symm_outflow = false, SGS_corrector = -1)


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
                back = [ back -SGS_corrector * reverse(vec[end-pad_size+1:end,i])]
                front = [ front SGS_corrector * everse(vec[1:pad_size,i])]
            else
                back = [ back (2*vec[end,i] .- 2*(vec[end,i]-b[i]) .- reverse(vec[end-pad_size+1:end,i]))]
                front = [ front reverse(vec[1:pad_size,i])]
            end
        elseif isnan(b[i]) && isnan(a[i]) == false
            #front = [ front a[i]*vec[end-pad_size+1:end,i].^0]
            if anti_symm_outflow
                back = [ back SGS_corrector * reverse(vec[end-pad_size+1:end,i])]
                front = [ front -SGS_corrector*reverse(vec[1:pad_size,i])]
            else
                back = [ back reverse(vec[end-pad_size+1:end,i])]
                front = [ front ((2*vec[1,i]) .-2*(vec[1,i]-a[i]) .-reverse(vec[1:pad_size,i]))]
            end
        else
            #front = [ front a[i]*vec[end-pad_size+1:end,i].^0]
            #back = [ back b[i]*vec[1:pad_size,i].^0]
            if anti_symm_outflow
                back = [ back -SGS_corrector*reverse(vec[end-pad_size+1:end,i])]
                front = [ front -SGS_corrector*reverse(vec[1:pad_size,i])]
            else
                back = [ back (2*vec[end,i] .- 2*(vec[end,i]-b[i]) .- reverse(vec[end-pad_size+1:end,i]))]
                front = [ front ((2*vec[1,i]) .-2*(vec[1,i]-a[i]) .-reverse(vec[1:pad_size,i]))]
            end
        end
    end
    return [front; vec ; back]
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



    #bar = gen_S_and_K(subgrid_filter_stencil,"just_avg")

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

    t_stencil = load(path * "T.jld")["t_stencil"]

    constraints,supply_s,dissipation,NN_descriptors,stencils,weight_bias,stencils = (load(path * "model.jld")[i] for i in ("constraints","supply_s","dissipation","NN_descriptors","stencils","weight_bias","stencils"))
    model, (conv,stencils) = gen_model(f, constraints,supply_s,dissipation,NN_descriptors,t_stencil,stencils = stencils)

    for i in 1:length(conv)
        conv[i].weight[:,:,:] = weight_bias[i][1]
        conv[i].bias[:] = weight_bias[i][2]
    end



    return model, (conv,stencils), t_stencil
end




using NNlib



function gen_model(f,constraints,supply_s,dissipation,NN_descriptors,t_stencil;stencils = 0)

    if constraints
        @assert constraints == supply_s  "If you use constraints you need to supply the SGS variables"
    end

    SGS_corrector = size(t_stencil)[1] * (t_stencil' * reverse(t_stencil))[1]


    (B,kernel_widths,channels,diagonals) = NN_descriptors

    @assert size(kernel_widths)[1] == size(channels)[1] + 1 "Supply n kernel_widths and n-1 channels"

    if stencils != 0
        stencil_width = Int((size(stencils[1])[1]-1)/2)
    end

    physics_width = Int((20-size(f(ones(20),ones(20),0))[1])/2)
    kernel_widths = [kernel_widths;]

    diagonals = 0

    if constraints
        if dissipation
            #channels = [[3]; channels; [2*diagonals*3 + 2*diagonals + 5]]
            channels = [[3]; channels; [4]]
        else
            channels = [[3]; channels; [2]]
        end
    else
        channels = [[3]; channels; [2]]
    end

    conv = conv_NN(2*kernel_widths .+ 1,channels,0)

    conv_pad_size = sum(kernel_widths)

    #B1,B2,B3 = 0,0,0
    if stencils == 0
        B1 = Float64.(Flux.glorot_uniform(Tuple(2*[B...] .+1)...,2,2))
        B2 = Float64.(Flux.glorot_uniform(Tuple(2*[B...] .+1)...,2,2))
        if dissipation
            B3 = Float64.(Flux.glorot_uniform(Tuple(2*[B...] .+1)...,2,2))
        else
            B3 = 0
        end
    else
        B1,B2,B3 = stencils
    end



    stencils = (B1,B2,B3)




    BC_f = padding_wrapper(f)



    model(u,s,x,as = 0,bs = 0,stencils = stencils,BC_f = BC_f,conv = conv,SGS_corrector = SGS_corrector,constraints = constraints,dissipation = dissipation,pad_sizes = (physics_width,B,conv_pad_size),supply_s = supply_s,diagonals = diagonals;channel = "all") = skew_symm_NN(u,s,x,as,bs,stencils,BC_f,conv,SGS_corrector,constraints,dissipation,pad_sizes,supply_s,diagonals;channel)

    return model, (conv,stencils)

end


function skew_symm_NN(u,s,x,as,bs,stencils,BC_f,conv,SGS_corrector,constraints,dissipation,pad_sizes,supply_s,diagonals;channel)

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
    pad3 = padding(s,conv_pad_size,as,bs,anti_symm_outflow = true,SGS_corrector = SGS_corrector)



    if supply_s
        pads = [pad1;pad2;pad3]
    else
        pads = [pad1;pad2; 0 .* pad3]
    end

    pads = reshape(pads,(size(pad1)[1],3,size(pad1)[2]))


    p = conv(pads)


    if constraints
        (B1,B2,B3) = stencils

        k = p[:,1:2,:]



        if dissipation
            q = p[:,3:4,:]
            #dTd = sum( d.^2 ,dims = [i for i in 1:dims])
        else
            q = 0
        end




        pad_u = padding(u,2*stencil_width + diagonals,as,bs)
        pad_s = padding(s,2*stencil_width + diagonals,0*as,0*bs,anti_symm_outflow = true,SGS_corrector = SGS_corrector)

        pads = [pad_u ;pad_s]

        a = reshape(pads,(size(pad_u)[1],2,size(pad_u)[2]))


        B1 = cons_mom_B(B1)
        B2 = cons_mom_B(B2)
        B3 = cons_mom_B(B3)


        B1_T = transpose_B(B1)

        B2_T = transpose_B(B2)
        B3_T = transpose_B(B3)




        cons = NNlib.conv(NNlib.conv(a,B1) .* k,B2_T) - NNlib.conv(NNlib.conv(a,B2) .* k,B1_T)
        if dissipation
            diss =  -NNlib.conv(q.^2 .* NNlib.conv(a,B3),B3_T)
        else
            diss = 0*cons
        end


        cons_u = cons[:,1,:]
        cons_s = cons[:,2,:]
        diss_u = diss[:,1,:]
        diss_s = diss[:,2,:]

        pu = cons_u + diss_u
        ps = cons_s + diss_s

    else
        pu = 1/(x[2]-x[1])* (circshift(p[:,1,:],(-1,0)) .- p[:,1,:])[2:end-1,:] #.*sp_mat_mul(p[:,1,:],[0,-1,1])
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

function cons_mom_B(B_kernel;channel = 1)
    if B_kernel != 0
        dims = length(size(B_kernel))-2
        channel_mask = gen_channel_mask(B_kernel,channel)

        means = mean(B_kernel,dims = collect(1:dims))
        return B_kernel .- means .* channel_mask
    else
        return 0
    end
end

function transpose_B(B_kernel)
    if B_kernel != 0
        dims = length(size(B_kernel))-2
        original_dims = stop_gradient() do
           collect(1:dims+2)
        end
        permuted_dims = stop_gradient() do
           copy(original_dims)
        end

        stop_gradient() do
            permuted_dims[dims+1] = original_dims[dims+2]
            permuted_dims[dims+2] = original_dims[dims+1]
        end

        T_B_kernel = permutedims(B_kernel,permuted_dims)

        for i in 1:dims
           T_B_kernel = reverse(T_B_kernel,dims = i)

        end

        return T_B_kernel
    else
        return 0
    end
end

function gen_channel_mask(data,channel)
    dims = length(size(data)) - 2
    number_of_channels = size(data)[end-1]
    channel_mask = stop_gradient() do
        zeros(size(data)[1:end-1])
    end
    stop_gradient() do
        channel_mask[[(:) for i in 1:dims]...,channel] .+= 1
    end
    return channel_mask
end
