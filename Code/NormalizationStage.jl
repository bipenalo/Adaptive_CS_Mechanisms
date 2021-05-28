# This computes the Normalization stage of the model

using LsqFit
using DifferentialEquations
using Sundials
using MAT
using JLD
using Plots
using IterableTables, DataFrames
using Distributions
pyplot()

# creating Gaussian function
function Gaussian(G, Amp, var, halfsize, fullsize)
    #fullsize = 400
    for i = 1:fullsize
        G[i] = Amp*exp(-(i - halfsize)*(i - halfsize)/var^2)
    end
    return G
end

# Spatiotemporal Input for channel 1
function SgnCh1(I, pad, delta, t, contrast)
    # Inside this function you can manually select the stimulus size and
    # the desired contrast value for the input signal. We picked this method
    # because it is the fastest.
    s = Int(20) # Stimulus size
    middle = pad + 1*1800
    base = 60.5 # background luminance
    # Select the contrast stimulus here is list. You can always use this formula
    # Delta = (MC * 2 * base)/(1 - MC) + base ; where MC is the Michelsen Contrast and base is the background luminance

    # Use this for a Rectangular Input
    I[1:(middle - s)] .= base
    if t >= 0.0 && t <= 1.0#4.0
        I[(middle - s):(middle + s)] .= contrast
    else
        I[(middle - s):(middle + s)] .= base
    end
    I[(middle + s):(end)] .= base

    # Use this for a Gaussian Input
    "I[1:(middle - s)] .= base
    if t >= 0.0 && t <= 4.0
        I[(middle - s):(middle + s)] = delta .+ base #.= contrast
    else
        I[(middle - s):(middle + s)] .= base
    end
    I[(middle + s):(end)] .= base"
    return I
end

# Spatiotemporal Input for channel 2
function SgnCh2(I, pad, delta, t, contrast)

    s = 279 # 392
    middle = pad + 1*1800 #- 3*50
    startL = middle - 2*s -48#+ 46 # marks the start of the left signal
    endL = startL + 2*s  # left signal
    startR = middle + 0*s +48#- 46# marks the start of the right signal
    endR = startR + 2*s

    base = 60.5
    # Delta = (MC * 2 * base)/(1 - MC) + base ; where MC is the Michelsen Contrast and base is the background luminance
    # Use this for a Rectangular Input
    I[1:startL] .= base
    I[endL:startR] .= base
    if t >= 0.5 && t <= 5.0
        I[startL:endL] .= contrast
        I[startR:endR] .= contrast
    else
        I[startL:endL] .= base
        I[startR:endR] .= base
    end
    I[endR:(end)] .= base # 7198 - 1799 = 5399

    # Use this for a Gaussian Input
    "I[1:(middle - s)] .= base
    if t >= 0.0 && t <= 4.0
        I[(middle - s):(middle + s)] = delta .+ base #.= contrast
    else
        I[(middle - s):(middle + s)] .= base
    end
    I[(middle + s):(end)] .= base"
    return I

end

# Input for the motion discrimination task
function SgnCh3(I, pad, delta, t, contrast)

    s = 350 # 392
    middle = pad + 1*1800 - 3*50
    startL = middle - 2*s + 46 # marks the start of the left signal
    endL = startL + 2*s  # left signal
    startR = middle + s -46# marks the start of the right signal
    endR = startR + 2*s

    base = 60.5

    #contrast = 1450.0000000000007 # change this depending on the contrast value
    I[1:startL] .= base #1799
    if t >= 0.0 && t <= 2.0
        I[startL:endR] .= contrast
    else
        I[startL:endR] .= base

    end
    I[endR:(end)] .= base # 7198 - 1799 = 5399
    # Use this for a Gaussian Input
    "I[1:(middle - s)] .= base
    if t >= 0.0 && t <= 4.0
        I[(middle - s):(middle + s)] = delta .+ base #.= contrast
    else
        I[(middle - s):(middle + s)] .= base
    end
    I[(middle + s):(end)] .= base"
    return I

end

# Main program starts here
function main()

    # Kernels
    n = 3599 # number of Kernel cells
    half_kernel = Int((n+1)/2)
    C_width = 2 # half of center's width
    S_width = 2 # half of surround's width
    Lc = half_kernel - C_width
    Rc = half_kernel + C_width
    Ls = Lc - S_width
    Rs = Rc + S_width
    Kc = zeros(n)
    Kc[Lc:Rc] .= 2.0
    Ks = zeros(n)
    Ks[Lc:Rc] .= 0.5
    Ks[Ls:Lc] .= 0.5
    Ks[Rc:Rs] .= 0.5
    Ks[1:Ls] .= 0.009
    Ks[Rs:n] .= 0.009
    DoK = Kc - Ks
    # build a gaussian input pattern
    fullsize = 1*3600
    halfsize = fullsize/2
    Gc = Array{Float64}(undef, fullsize)
    f = 1.24
    s = 20 # 105
    RFc_gaussian = Gaussian(Gc, 3.48, f*s, halfsize, fullsize) # (2.8% = 3.48; 92% = 1389.5)
    deltaGaussian = RFc_gaussian[(1800-Int(3*s)):(1800+Int(3*s))] # 1 std = 1735:1865; sm = 1780:1820
    # Initial Conditions of the system
    L0 = 60.5
    I = L0*ones(fullsize) # note: the signal starts at position i = 7

    # Parameters of the equation
    A, B, D = 0.5, 1.0, 0.0
    # initial conditions for neuron layer x
    I0 = L0*sum(Kc) # first value after zero padding
    Ij = L0*sum(Ks)


    X0 = B*I0/(A + I0 + Ij)
    println(X0)
    MC = 0.028 # Michelsen Contrast
    deltaI = (MC * 2 * L0)/(1 - MC) + L0
    println(deltaI)

    noiseInc = 0.5 # % noise increment
    noiselessValue = 0.2266 # value without noise
    noise = noiselessValue * noiseInc
    #println(noise)


    # Defining the function
    function f(du, u, p, t)
        A, B, RFc, RFs, Inp, numcell, filter, pad, deltaGaussian, DeltaI, Noise = p

        println(t)
        # change SgnCh1 to SgnCh2 if second channels is to be model
        I = SgnCh1(Inp, pad, deltaGaussian, t, DeltaI) # call the input value at time t

        ###### Middle of the x cell array ######
        for i = 1:numcell
            #println(i)
            esum, isum = 0.0, 0.0
            esum = sum(I[i:i+filter-1].*RFc)
            isum = sum(I[i:i+filter-1].*RFs)
            du[i] = -A*u[i] + (B - u[i])*esum - u[i]*isum
        end

    end # end of function f

    function Gnoise(du, u, p, t)
        A, B, RFc, RFs, Inp, numcell, filter, pad, deltaGaussian, DeltaI, Noise = p
        for i = 1:3600
            du[i] = Noise
        end
    end

    # ODE
    x0 = Array{Float64}(undef, (fullsize))
    # fill array with initial values
    for i =1:fullsize
        x0[i] = X0
    end


    # Input Stimuli S
    Input_1 = zeros(1*3600) #arr_46
    input_r = size(Input_1)[1]

    # Zero padding
    filter = size(Kc)[1]  # Size of the Kernel
    pad = (filter - 1) ÷ 2 # Integer division. Number of zeros to be added at both edges.
    input_padded = zeros(input_r+(2*pad))

    Input_1 = input_padded
    input_r = size(Input_1)[1]
    result = zeros(input_r-2*pad) # Size of the output signal

    result_r = size(result)[1]
    display(input_r)
    p = [A, B, Kc, Ks, Input_1, result_r, filter, pad, deltaGaussian, deltaI, noise]
    tspan = (0.0,2.0) #6.0
    prob = ODEProblem(f,x0,tspan,p) # solver
    #prob = SDEProblem(f, Gnoise, x0,tspan,p) # uncomment if noise is to be added
    sol = solve(prob, alg_hints=[:stiff]) # use :stiff for stiff problems
    #sol = solve(prob, alg_hints=[:additive], reltol=1e-1,abstol=1e-1) # use :stiff for stiff problems and additive to indicate addtive noise



    return sol # main return
end

out = @time main() # run the main function
times = [t for t in out.t] # total number of time steps
fullsize = 1*3600
xi = Array{Float64}(undef, (fullsize, length(times)))

for i = 1:fullsize
    xi[i, :] = [u[i] for u in out.u] # for each cell i store all temporal stamps (:)
end

# select which instance of the model you'd like to save viz. channel 1 or channel 2
ch = "ch1"
if ch == "ch1"
    save("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Noise/Noise28_sm_50_time.jld", "xi_ss", times)
    println(size(xi))
    println(size(xi))
    println(xi[1800, trunc(Int,length(times)/2)])
    println(xi[980, trunc(Int,length(times)/2)])
    display(plot(times, xi[1800, :]))
    display(plot(xi[:, trunc(Int,length(times)/2)]))
else
    save("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/MStas/Trial2/contrast22_MStask_ch2.jld", "xi_ss",  xi)
    save("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/MStask/Trial2/contrast22_time_MStask_ch2.jld", "xi_ss", times)
    println(size(xi))
    println(xi[1600, trunc(Int,length(times)/2)])
    println(xi[980, trunc(Int,length(times)/2)])
    display(plot(times, xi[1600, :]))
    display(plot(xi[:, trunc(Int,length(times)/2)]))
end

# save the file into .mat format
filem = matopen("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Noise/Noise28_sm_50.mat", "w")
write(filem, "noise28_sm_50", xi)
close(filem)

file_time = matopen("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Noise/Noise28_sm_50_time.mat", "w")
write(file_time, "noise28_sm_50_time", times)
close(file_time)
