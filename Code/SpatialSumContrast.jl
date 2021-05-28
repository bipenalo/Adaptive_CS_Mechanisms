# testing the effect of contrast on spatial summation 
# Final model
# Fine-tuning the new model which is luminance-invariant
using LsqFit
using DifferentialEquations
using Distributions
using Sundials
using JLD
using Plots
using IterableTables, DataFrames, GLM
using CSV
pyplot()


# creating Gaussian function
function Gaussian(G, Amp, var, halfsize, fullsize)
    #fullsize = 400
    for i = 1:fullsize
        G[i] = Amp*exp(-1*(i - halfsize)*(i - halfsize)/var^2)
    end
    return G
end


# Sigmoid Non-linearity for the center
function SigmoidC(x, c)
    b, center, amp = [75.251, 0.261, 7.] # 75.251, 0.261, 7
    xb = 0.22#0.22
    if x < xb
        sig = 0.23 #0.23
    elseif c == 1 && x >= xb
        sig = amp/(1 + exp(-b*(x - center  )))
    elseif c == 2 && x >= xb
        sig = amp/(1 + exp(-b*(x - center  )))
    elseif c == 3 && x >= xb
        sig = amp/(1 + exp(-b*(x - center  )))
    elseif c == 4 && x >= xb
        sig = amp/(1 + exp(-b*(x - center  )))
    elseif c == 5 && x >= xb
        sig = amp/(1 + exp(-b*(x - center  )))
    elseif c == 6 && x >= xb
        sig = amp/(1 + exp(-b*(x - center  )))
    end
    return sig
end

# Sigmoid Non-linearity for the surround
function SigmoidS(x, c)
    b, center, amp = [80.7925, 0.2685, 6.5] # 80.7925, 0.2685, 6.5
    xb = 0.233#0.233
    if x < xb
        sig = 0.253  #0.253
    elseif c == 1 && x >= xb
        sig = amp/(1 + exp(-b *(x - center)))
    elseif c == 2 || c == 3 && x >= xb #|| x < 0.22
        sig = amp/(1 + exp(-b *(x - center)))
    elseif c == 4 && x >= xb
        sig = amp/(1 + exp(-b*(x - center  )))
    elseif c == 5 && x >= xb
        sig = amp/(1 + exp(-b*(x - center  )))
   elseif c == 6 && x >= xb
        sig = amp/(1 + exp(-b*(x - center  )))
    end
    return sig
end

# Input
function Inp_norm(I, S, contrast, base, middle)

     if S == 1
        size = 2 #5
        I[1:(middle - size)] .= base
        I[(middle - size):(middle + size)] .= contrast # 0.63 deg
        I[(middle + size):end] .= base
    elseif S == 2
        size = 5 #5
        I[1:(middle - size)] .= base
        I[(middle - size):(middle + size)] .= contrast # 0.63 deg
        I[(middle + size):end] .= base
    elseif S == 3
        size = 10 #5
        I[1:(middle - size)] .= base
        I[(middle - size):(middle + size)] .= contrast # 0.63 deg
        I[(middle + size):end] .= base
    elseif S == 4
        size = 15 #5
        I[1:(middle - size)] .= base
        I[(middle - size):(middle + size)] .= contrast # 0.63 deg
        I[(middle + size):end] .= base
    elseif S == 5
        size = 20 #20
        I[1:(middle - size)] .= base
        I[(middle - size):(middle + size)] .= contrast # 1.31 deg
        I[(middle + size):end] .= base
    elseif S == 6
        size = 28 #28
        I[1:(middle - size)] .= base
        I[(middle - size):(middle + size)] .= contrast # 2.64 deg
        I[(middle + size):end] .= base
    elseif S == 7
        size = 37 # 37
        I[1:(middle - size)] .= base
        I[(middle - size):(middle + size)] .= contrast # 4.00 deg
        I[(middle + size):end] .= base
    elseif S == 8
        size = 45#45
        I[1:(middle - size)] .= base
        I[(middle - size):(middle + size)] .= contrast # 5.00 deg
        I[(middle + size):end] .= base
    elseif S == 9
        size = 55#45
        I[1:(middle - size)] .= base
        I[(middle - size):(middle + size)] .= contrast # 5.00 deg
        I[(middle + size):end] .= base
    elseif S == 10
        size = 65#45
        I[1:(middle - size)] .= base
        I[(middle - size):(middle + size)] .= contrast # 5.00 deg
        I[(middle + size):end] .= base
    elseif S == 11
        size = 85#45
        I[1:(middle - size)] .= base
        I[(middle - size):(middle + size)] .= contrast # 5.00 deg
        I[(middle + size):end] .= base
    elseif S == 12
        size = 105 #105
        I[1:(middle - size)] .= base
        I[(middle - size):(middle + size)] .= contrast # 1.31 deg
        I[(middle + size):end] .= base
    elseif S == 13
        size = 160 #160
        I[1:(middle - size)] .= base
        I[(middle - size):(middle + size)] .= contrast # 2.64 deg
        I[(middle + size):end] .= base
    else
        size = 205 # 205
        I[1:(middle - size)] .= base
        I[(middle - size):(middle + size)] .= contrast # 4.00 deg
        I[(middle + size):end] .= base
    end
    return I

end



u = 14
# building the input stage after normalization for simplicity
kernel_fullsize = 3600
half = Int(kernel_fullsize/2)
I = zeros(kernel_fullsize)

arr_11 = Array{Float64}(undef, (kernel_fullsize, u))
v_11 = [0.251, 0.251, 0.251, 0.251, 0.251, 0.251, 0.251, 0.251, 0.251, 0.251, 0.251, 0.251, 0.251, 0.251]
b11 = [0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.217, 0.2176, 0.2164, 0.217, 0.2176, 0.2164, 0.2155, 0.2147]

arr_22 = Array{Float64}(undef, (kernel_fullsize, u))
v_22 = [0.2922, 0.2922, 0.2922, 0.2922, 0.2922, 0.2922, 0.2922, 0.2922, 0.2922, 0.2922, 0.2914, 0.2884, 0.286, 0.285] #0.2922, 0.2914, 0.2884, 0.286, 0.285
b22 = [0.21, 0.21, 0.21, 0.21, 0.21, 0.21, 0.2176, 0.2169, 0.2143, 0.2176, 0.2169, 0.2143, 0.2122, 0.2100]

for n=1:u
    arr_11[:, n] = Inp_norm(I, n, v_11[n], b11[n], half)
    arr_22[:, n] = Inp_norm(I, n, v_22[n], b22[n], half)
end

# Store the matrixes in the respective containers
Data = Array{Float64}(undef, (size(arr_11)[1], size(arr_11)[2], 2)) # create a cell struct.. with 5 contrast
Data[:,:,1] = arr_11
Data[:,:,2] = arr_22

# Determining the Gaussian RFs
fullsize = 3600
halfsize = fullsize/2
Gc = Array{Float64}(undef, fullsize)
Gs = Array{Float64}(undef, fullsize)
xi_ss = Array{Float64}(undef, (fullsize, u)) # create a 400 x 3 array
y_ss = Array{Float64}(undef, u) # create an array to store the ss of the output cell y.
x200_ss = Array{Float64}(undef, u) # create an array to store the ss of the output cell y.
RFc = Gaussian(Gc, 2/1, 67.0, halfsize, fullsize) # (2 and 67)
RFs = Gaussian(Gs, 1/1, 300.0, halfsize, fullsize) # (1 and 300)
Dog = RFc - RFs
#plot(Dog, linewidth=8, grid=false)

plot(arr_11[:, 1], label="0.63 deg", xlab="space")
plot!(arr_11[:, 2], label="1.31 deg", xlab="space")
plot!(arr_11[:, 3], label="2.64 deg", xlab="space", linewidth=1)
plot!(arr_11[:, 4], label="4 deg", xlab="space")
display(plot!(arr_11[:, 8], label="5 deg", xlab="space", linewidth=1))
plot!((0.025*RFc .+0.218), label="RFc", xlab="space")
plot!(0.025*RFs .+0.218, label="RFs", xlab="space")
display(plot!((0.025*Dog.+0.218), label="DoG"))


y = Array{Float64}(undef, (u, 2)) # array to store the steady state values
# Array{Float64,N} where N(::UndefInitializer, ::Int64)
time_arr = Array{Float64}[] # array of any dims.
x_arr = Array{Float64}[] # array of any dims.

ratios = Array{Float64}(undef, (u, 2))
Inh_act = Array{Float64}(undef, (u, 2))
Exc_act = Array{Float64}(undef, (u, 2))
contraste = ["2.8%", "5.5%", "11%", "22%", "46%", "92%"]
for j=1:2 # for loop over the Data struct of contrast values
    fullsize = 3600
    # for loop to calculate Yex
    Yex = Array{Float64}(undef, (fullsize, 14))
    Yinh = Array{Float64}(undef, (fullsize, 14))

    for i = 1:fullsize
        # exitation
        Yex[i, 1] = RFc[i] * SigmoidC(Data[i, 1, j], j) # size = 0.63 deg
        Yex[i, 2] = RFc[i] * SigmoidC(Data[i, 2, j], j) # size = 1.31 deg
        Yex[i, 3] = RFc[i] * SigmoidC(Data[i, 3, j], j) # size = 2.64 deg
        Yex[i, 4] = RFc[i] * SigmoidC(Data[i, 4, j], j) # size = 4 deg
        Yex[i, 5] = RFc[i] * SigmoidC(Data[i, 5, j], j) # size = 5 deg
        Yex[i, 6] = RFc[i] * SigmoidC(Data[i, 6, j], j) # size = 2.64 deg
        Yex[i, 7] = RFc[i] * SigmoidC(Data[i, 7, j], j) # size = 4 deg
        Yex[i, 8] = RFc[i] * SigmoidC(Data[i, 8, j], j) # size = 5 deg
        Yex[i, 9] = RFc[i] * SigmoidC(Data[i, 9, j], j) # size = 5 deg
        Yex[i, 10] = RFc[i] * SigmoidC(Data[i, 10, j], j) # size = 5 deg
        Yex[i, 11] = RFc[i] * SigmoidC(Data[i, 11, j], j) # size = 5 deg
        Yex[i, 12] = RFc[i] * SigmoidC(Data[i, 12, j], j) # size = 5 deg
        Yex[i, 13] = RFc[i] * SigmoidC(Data[i, 13, j], j) # size = 5 deg
        Yex[i, 14] = RFc[i] * SigmoidC(Data[i, 14, j], j) # size = 5 deg

        # inhibition
        Yinh[i, 1] = RFs[i] * SigmoidS(Data[i, 1, j], j)
        Yinh[i, 2] = RFs[i] * SigmoidS(Data[i, 2, j], j)
        Yinh[i, 3] = RFs[i] * SigmoidS(Data[i, 3, j], j)
        Yinh[i, 4] = RFs[i] * SigmoidS(Data[i, 4, j], j)
        Yinh[i, 5] = RFs[i] * SigmoidS(Data[i, 5, j], j)
        Yinh[i, 6] = RFs[i] * SigmoidS(Data[i, 6, j], j)
        Yinh[i, 7] = RFs[i] * SigmoidS(Data[i, 7, j], j)
        Yinh[i, 8] = RFs[i] * SigmoidS(Data[i, 8, j], j)
        Yinh[i, 9] = RFs[i] * SigmoidS(Data[i, 9, j], j)
        Yinh[i, 10] = RFs[i] * SigmoidS(Data[i, 10, j], j)
        Yinh[i, 11] = RFs[i] * SigmoidS(Data[i, 11, j], j)
        Yinh[i, 12] = RFs[i] * SigmoidS(Data[i, 12, j], j)
        Yinh[i, 13] = RFs[i] * SigmoidS(Data[i, 13, j], j)
        Yinh[i, 14] = RFs[i] * SigmoidS(Data[i, 14, j], j)
    end
    Act_Ex = sum(Yex, dims=1)  # sums the activity of Yex
    Act_Inh = sum(Yinh, dims=1) #
    println("Contrast", contraste[j])
    println("esum: ", Act_Ex)
    println("isum: ", Act_Inh)
    ratios[:, j] = Act_Ex ./ Act_Inh
    Inh_act[:, j] = Act_Inh
    Exc_act[:, j] = Act_Ex


    # Main function
    function main(Ex, Inh, y0)

        function f(du, u, p, t)
            A, B, D, YsumEx, YsumInh = p
            # multiplicative membrane equation
            du[1] = -A*u[1] + (B - u[1]) * YsumEx - (D + u[1]) * YsumInh

        end

        A, B, D, yEx, yInh = 1, 100, 50, Ex, Inh
        #display(Ex)
        p = [A, B, D, yEx, yInh] # Vector
        tspan = (0.0,4.0)
        prob = ODEProblem(f,[y0,],tspan,p) # no noise
        sol = solve(prob, alg_hints=[:stiff])
        return sol
    end # end of Main function


    fullsize = 3600
    # Initial Conditions of the system
    L0 = 60.5
    I = L0*ones(fullsize) # note: the signal starts at position i = 7
    sumGc = sum(RFc)
    sumGs = sum(RFs)
    # Parameters of the equation
    B, A, By, Dy = 1, 1, 100.0, 50.0
    # initial conditions for neuron layer x
    I0 = L0*10 # first value after zero padding
    Ij = L0*35.837#8.837 35.837
    X0 = B*I0/(0.5 + I0 + Ij)
    println(X0)
    # initial conditions for neuron layer y
    Y0 = (By * sumGc * 0.23 - Dy * sumGs * 0.253)/(A + sumGc * 0.23 + sumGs * 0.253)

    println(Y0)
    # loop over the 14 stimulus sizes
    for k=1:14
        out = main(Act_Ex[k],Act_Inh[k], Y0)
        # Defining the function
        xi = [u[1] for u in out.u]
        t = [t for t in out.t]
        #display(plot(t, xi))
        push!(time_arr, t) # the push! function concatenates an array of any dim to the bottom of the matrix.
        push!(x_arr, xi)
        y[k, j] = xi[end] # get the steady state value of xi
    end


end # Data struc loop
display(y)


sizesL = [0.063, 0.16, 0.31, 0.47, 0.63, 0.89, 1.15, 1.31,1.7, 2.04, 2.3, 2.64, 4, 5]

F = plot(sizesL, y[:, 1], ylabel="Membrane potential", grid=false,  xlabel= "Size (deg)", label="10%", linewidth=2.5 ,xtickfontsize= 13, ytickfontsize=13, yguidefontsize= 16, xguidefontsize= 16, shape=:utriangle, markersize=10, color=:red, markercolor=:red, legendfont=font(12))

display(plot!(sizesL, y[:, 2], ylabel="Membrane potential", xlabel= "Size (deg)", label="28%", linewidth=2.5, color=:orange, shape=:circle, markersize=10, markercolor=:orange))
#savefig(F, "/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/reviewer2_nSS")

# for plots visit = http://julia.cookbook.tips/doku.php?id=plotattributes#plot_title
