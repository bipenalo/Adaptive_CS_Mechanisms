# noise analysis
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
        G[i] = Amp*exp(-(i - halfsize)*(i - halfsize)/var^2)
    end
    return G
end

function chiqtest(O, E, sem, n)
    stdv = sem * sqrt(n)
    variance = stdv.^2
    x2 = sum((O - E).^2 ./ variance)
    return x2
end

function SSqs(f, obs)
    average = mean(obs)
    n = size(obs)[1] # tuple converted to integer
    avg_arr = ones(n)*average
    SS_tot = sum((obs - avg_arr).^2)
    SS_res = sum((obs - f).^2)

    return SS_tot, SS_res
end

# Sigmoid Non-linearity for the center
function SigmoidC(x, c)
    b, center, amp = [75.251, 0.261, 7.] # 75.251, 0.261, 7
    xb = 0.22
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
    b, center, amp = [80.7925, 0.2685, 6.5] # 78.7925, 0.2685, 6.5
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
        size = 20 #20
        I[1:(middle - size)] .= base
        I[(middle - size):(middle + size)] .= contrast # 0.63 deg
        I[(middle + size):end] .= base
    elseif S == 2
        size = 45 #45
        I[1:(middle - size)] .= base
        I[(middle - size):(middle + size)] .= contrast # 1.31 deg
        I[(middle + size):end] .= base
    elseif S == 3
        size = 105 #105
        I[1:(middle - size)] .= base
        I[(middle - size):(middle + size)] .= contrast # 2.64 deg
        I[(middle + size):end] .= base
    elseif S == 4
        size = 160 # 160
        I[1:(middle - size)] .= base
        I[(middle - size):(middle + size)] .= contrast # 4.00 deg
        I[(middle + size):end] .= base
    else
        size = 205#205
        I[1:(middle - size)] .= base
        I[(middle - size):(middle + size)] .= contrast # 5.00 deg
        I[(middle + size):end] .= base
    end
    return I

end


# building the input stage after normalization for simplicity
kernel_fullsize = 3600
half = Int(kernel_fullsize/2)
I = zeros(kernel_fullsize)
arr_2_8 = Array{Float64}(undef, (kernel_fullsize, 5))
v_2_8 = [0.2266, 0.2266, 0.2267, 0.2270, 0.2272]
b28 = [ 0.2181, 0.2181, 0.2181, 0.2181, 0.2181]
#b28 = [ 0.19, 0.19, 0.19, 0.19, 0.19]
arr_5_5 = Array{Float64}(undef, (kernel_fullsize, 5))
v_5_5 = [0.2354, 0.2350, 0.2348, 0.2342, 0.2332]
b55 = [0.218, 0.2176, 0.2175, 0.2169, 0.2159]
arr_11 = Array{Float64}(undef, (kernel_fullsize, 5))
v_11 = [0.249, 0.2531, 0.2518, 0.2508, 0.2497]
b11 = [0.217, 0.2176, 0.2164, 0.2155, 0.2147]
arr_22 = Array{Float64}(undef, (kernel_fullsize, 5))
v_22 = [0.2922, 0.2914, 0.2884, 0.286, 0.285]
b22 = [0.2176, 0.2169, 0.2143, 0.2122, 0.2100]
arr_46 = Array{Float64}(undef, (kernel_fullsize, 5))
v_46 = [0.392, 0.3790, 0.359, 0.343, 0.3574]
b46 = [0.216, 0.196, 0.189, 0.175, 0.173]
arr_92 = Array{Float64}(undef, (kernel_fullsize, 5))
v_92 = [0.6684, 0.6571, 0.612, 0.59, 0.554]
b92 = [0.20, 0.1779, 0.1414, 0.1024, 0.08]

conT = ["28_", "92_"]
size1 = ["sm_", "m_", "lg_"]
noiseL = ["30.jld", "10.jld", "05.jld"]

# load inputs from the noisy data obtained with the NormalizationStage.jl
In1 = load("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Noise/Noise28_sm_30.jld")
noise28_sm_30 = In1["xi_ss"]
In2 = load("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Noise/Noise28_sm_10.jld")
noise28_sm_10 = In2["xi_ss"]
In3 = load("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Noise/Noise28_sm_05.jld")
noise28_sm_05 = In3["xi_ss"]
In4 = load("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Noise/Noise28_m_30.jld")
noise28_m_30 = In4["xi_ss"]
In5 = load("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Noise/Noise28_m_10.jld")
noise28_m_10 = In5["xi_ss"]
In6 = load("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Noise/Noise28_m_05.jld")
noise28_m_05 = In6["xi_ss"]
In7 = load("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Noise/Noise28_lg_30.jld")
noise28_lg_30 = In7["xi_ss"]
In8 = load("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Noise/Noise28_lg_10.jld")
noise28_lg_10 = In8["xi_ss"]
In9 = load("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Noise/Noise28_lg_05.jld")
noise28_lg_05 = In9["xi_ss"]

DataNoise = Array{Float64}(undef, (3600, 9, 1)) # create a cell structuew with 3600 neurons, 9 sizes, 3 noise levels evaluated at one contrast value (2.8%)
DataNoise[:, 1, 1] = noise28_sm_30[:, trunc(Int, (size(noise28_sm_30)[2])/2)]
DataNoise[:, 2, 1] = noise28_m_30[:, trunc(Int, (size(noise28_m_30)[2])/2)]
DataNoise[:, 3, 1] = noise28_lg_30[:, trunc(Int, (size(noise28_lg_30)[2])/2)]
DataNoise[:, 4, 1] = noise28_sm_10[:, trunc(Int, (size(noise28_sm_10)[2])/2)]
DataNoise[:, 5, 1] = noise28_m_10[:, trunc(Int, (size(noise28_m_10)[2])/2)]
DataNoise[:, 6, 1] = noise28_lg_10[:, trunc(Int, (size(noise28_lg_10)[2])/2)]
DataNoise[:, 7, 1] = noise28_sm_05[:, trunc(Int, (size(noise28_sm_05)[2])/2)]
DataNoise[:, 8, 1] = noise28_m_05[:, trunc(Int, (size(noise28_m_05)[2])/2)]
DataNoise[:, 9, 1] = noise28_lg_05[:, trunc(Int, (size(noise28_lg_05)[2])/2)]

# Determining the Gaussian RFs
fullsize = 3600
halfsize = fullsize/2
Gc = Array{Float64}(undef, fullsize) # center Gaussian
Gs = Array{Float64}(undef, fullsize) # surround Gaussian
xi_ss = Array{Float64}(undef, (fullsize, 5)) # create a 400 x 3 array
y_ss = Array{Float64}(undef, 5) # create an array to store the ss of the output cell y.
RFc = Gaussian(Gc, 2/1, 67.0, halfsize, fullsize) # (2 and 67)
RFs = Gaussian(Gs, 1/1, 300.0, halfsize, fullsize) # (1 and 300)
Dog = RFc - RFs
#plot(Dog, linewidth=8, grid=false)

plot(arr_2_8[:, 1], label="0.63 deg", xlab="space")
plot!(arr_2_8[:, 2], label="1.31 deg", xlab="space")
plot!(arr_2_8[:, 3], label="2.64 deg", xlab="space", linewidth=1)
plot!(arr_2_8[:, 4], label="4 deg", xlab="space")
display(plot!(arr_2_8[:, 5], label="5 deg", xlab="space", linewidth=1))
plot!((0.025*RFc .+0.218), label="RFc", xlab="space")
plot!(0.025*RFs .+0.218, label="RFs", xlab="space")
display(plot!((0.025*Dog.+0.218), label="DoG"))


y = Array{Float64}(undef, (9, 1)) # array to store the steady state values
time_arr = Array{Float64}[] # array of any dims.
x_arr = Array{Float64}[] # array of any dims.

contraste = ["2.8%", "5.5%", "11%", "22%", "46%", "92%"]
for j=1:1 # for loop over the Data struct of contrast values
    fullsize = 3600
    # for loop to calculate Yex
    Yex = Array{Float64}(undef, (fullsize, 9))
    Yinh = Array{Float64}(undef, (fullsize, 9))

    for i = 1:fullsize
        # exitation
        Yex[i, 1] = RFc[i] * SigmoidC(DataNoise[i, 1, j], j) # 30% size = 0.63 deg
        Yex[i, 2] = RFc[i] * SigmoidC(DataNoise[i, 2, j], j) # size = 2.64 deg
        Yex[i, 3] = RFc[i] * SigmoidC(DataNoise[i, 3, j], j) # size = 5 deg
        Yex[i, 4] = RFc[i] * SigmoidC(DataNoise[i, 4, j], j) # 10%
        Yex[i, 5] = RFc[i] * SigmoidC(DataNoise[i, 5, j], j)
        Yex[i, 6] = RFc[i] * SigmoidC(DataNoise[i, 6, j], j)
        Yex[i, 7] = RFc[i] * SigmoidC(DataNoise[i, 7, j], j) # 5%
        Yex[i, 8] = RFc[i] * SigmoidC(DataNoise[i, 8, j], j)
        Yex[i, 9] = RFc[i] * SigmoidC(DataNoise[i, 9, j], j)
        # inhibition
        Yinh[i, 1] = RFs[i] * SigmoidS(DataNoise[i, 1, j], j)# 30%
        Yinh[i, 2] = RFs[i] * SigmoidS(DataNoise[i, 2, j], j)
        Yinh[i, 3] = RFs[i] * SigmoidS(DataNoise[i, 3, j], j)
        Yinh[i, 4] = RFs[i] * SigmoidS(DataNoise[i, 4, j], j) # 10%
        Yinh[i, 5] = RFs[i] * SigmoidS(DataNoise[i, 5, j], j)
        Yinh[i, 6] = RFs[i] * SigmoidS(DataNoise[i, 6, j], j)
        Yinh[i, 7] = RFs[i] * SigmoidS(DataNoise[i, 7, j], j) # 5%
        Yinh[i, 8] = RFs[i] * SigmoidS(DataNoise[i, 8, j], j)
        Yinh[i, 9] = RFs[i] * SigmoidS(DataNoise[i, 9, j], j)
    end
    Act_Ex = sum(Yex, dims=1)  # sums the activity of Yex
    Act_Inh = sum(Yinh, dims=1) #
    println("Contrast", contraste[j])
    println("esum: ", Act_Ex)
    println("isum: ", Act_Inh)



    # Main function
    function main(Ex, Inh, y0)

        function f(du, u, p, t)
            A, B, D, YsumEx, YsumInh = p
            du[1] = -A*u[1] + (B - u[1]) * YsumEx - (D + u[1]) * YsumInh
        end

        function Gnoise(du, u, p, t)
            du[1] = 10.0 # noise amplitude
        end

        A, B, D, yEx, yInh = 1, 100, 50, Ex, Inh
        #display(Ex)
        p = [A, B, D, yEx, yInh] # Vector
        tspan = (0.0,4.0)
        #prob = ODEProblem(f,[y0,],tspan,p)
        prob = SDEProblem(f, Gnoise, [y0,],tspan,p) # stochastic differential equation

        sol = solve(prob, alg_hints=[:stiff])
        return sol
    end # end of Main function


    fullsize = 3600
    # Initial Conditions of the system
    L0 = 60.5
    I = L0*ones(fullsize)
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
    #println(sumGc)
    # loop over the nine sizes
    for k=1:9
        out = main(Act_Ex[k],Act_Inh[k], Y0)
        # Defining the function
        xi = [u[1] for u in out.u]
        t = [t for t in out.t]
        display(plot(t, xi))
        push!(time_arr, t) # the push! function concatenates an array of any dim to the bottom of the matrix.
        push!(x_arr, xi)
        y[k, j] = xi[end] # get the steady state value of xi
    end


end # Data struc loop
display(y)


# plotting the esum/isum ratio as a function of Size
sizes = [0.63, 1.31, 2.64, 4, 5]
Scale = 218.47


Threshold = 1 ./ y
display(Threshold)
scatter([0.63, 2.64, 5], Scale *Threshold[1:3, 1], label="30%", markershape =:star4, markersize=9, markercolor=:blue)
scatter!([0.63, 2.64, 5], Scale *Threshold[4:6, 1], label="10%", markershape =:utriangle, markersize=9, markercolor=:yellow)
display(scatter!([0.63, 2.64, 5], Scale *Threshold[7:9, 1], label="5%", markershape =:rect, markersize=8, markercolor=:green))

# load model's output (noiseless)
Output = load("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Noise/ModelOutput.jld")
modeloutput = Output["xi_ss"]



# Tadin's data
contrast2_8 = DataFrame!(CSV.File("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Adative_CS/Plots/Tadin's data/2_8_contrast.csv", datarow=1))
contrast5_5 = DataFrame!(CSV.File("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Adative_CS/Plots/Tadin's data/5_5_contrast.csv", datarow=1))
contrast11 = DataFrame!(CSV.File("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Adative_CS/Plots/Tadin's data/11_contrast.csv", datarow=1))
contrast22 = DataFrame!(CSV.File("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Adative_CS/Plots/Tadin's data/22_contrast.csv", datarow=1))
contrast46 = DataFrame!(CSV.File("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Adative_CS/Plots/Tadin's data/46_contrast.csv", datarow=1))
contrast92 = DataFrame!(CSV.File("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Adative_CS/Plots/Tadin's data/92_contrast.csv", datarow=1))
# Tadin's data (std)
contrast2_8_std = DataFrame!(CSV.File("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Adative_CS/Plots/Tadin's data/Tadin_std_2_8.csv", datarow=1))
select!(contrast2_8_std, Not(2))
contrast5_5_std = DataFrame!(CSV.File("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Adative_CS/Plots/Tadin's data/Tadin_std_5_5.csv", datarow=1))
select!(contrast5_5_std, Not(2))
contrast11_std = DataFrame!(CSV.File("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Adative_CS/Plots/Tadin's data/Tadin_std_11.csv", datarow=1))
select!(contrast11_std, Not(2))
contrast22_std = DataFrame!(CSV.File("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Adative_CS/Plots/Tadin's data/Tadin_std_22.csv", datarow=1))
select!(contrast22_std, Not(2))
contrast46_std = DataFrame!(CSV.File("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Adative_CS/Plots/Tadin's data/Tadin_std_46.csv", datarow=1))
select!(contrast46_std, Not(2))
contrast92_std = DataFrame!(CSV.File("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Adative_CS/Plots/Tadin's data/Tadin_std_92.csv", datarow=1))
select!(contrast92_std, Not(2))
error_2_8 = contrast2_8_std[!, 2] .- contrast2_8[!, 2]
error_5_5 = contrast5_5_std[!, 2] .- contrast5_5[!, 2]
error_11 = contrast11_std[!, 2] .- contrast11[!, 2]
error_22 = contrast22_std[!, 2] .- contrast22[!, 2]
error_46 = contrast46_std[!, 2] .- contrast46[!, 2]
error_92 = contrast92_std[!, 2] .- contrast92[!, 2]

h1 = plot(contrast2_8[!, 1], contrast2_8[!, 2], ribbon = error_2_8, fillalpha=0.4, title="2.8%", legend=false, titlefontsize=14, grid=false, xlims=(0.25, 5.5), color=:salmon, linewidth=1.5, yscale=:log2, markershape =:circle, markersize=7,yguidefontsize=13, xtickfontsize= 12, ytickfontsize=12,  markercolor=:salmon, markerstrokecolor=:black)
yticks!([5, 10, 20, 40, 60, 90], ["5", "10", "20", "40", "60", "90"])
plot!(sizes, Scale * modeloutput[:, 1], yaxis=:log2, xlims=(0.25, 5.5), linewidth=2, color=:black)
scatter!([0.63, 2.64, 5], Scale *Threshold[1:3, 1], label="30%", markershape =:star4, markersize=9, markercolor=:blue)
scatter!([0.63, 2.64, 5], Scale *Threshold[4:6, 1], label="10%", markershape =:utriangle, markersize=9, markercolor=:yellow)
h6 = display(scatter!([0.63, 2.64, 5], Scale *Threshold[7:9, 1], label="5%", markershape =:rect, markersize=8, markercolor=:green))

h2 = plot(contrast5_5[!, 1], contrast5_5[!, 2], ribbon=error_5_5, fillalpha=0.4, title="5.5%", legend=false, titlefontsize=14, grid=false, xlims=(0.25, 5.5), linewidth=1.5, yscale=:log2, markershape =:circle, markersize=7,xtickfontsize= 12, ytickfontsize=12,  color=:violet, markercolor=:violet, markerstrokecolor=:black)
yticks!([5, 10, 20, 40, 60, 90], ["5", "10", "20", "40", "60", "90"])
h7 = display(plot!(sizes, Scale * modeloutput[:, 2], yaxis=:log2, xlims=(0.25, 5.5), linewidth=2, color=:black))

h3 = plot(contrast11[!, 1], contrast11[!, 2], ribbon=error_11, fillalpha=0.4, grid=false, legend=false, xlims=(0.25, 5.5), title="11%", titlefontsize=14, linewidth=1.5, yscale=:log2, markershape =:hexagon, markersize=7,xtickfontsize= 12, ytickfontsize=12,  color=:green, markercolor=:green, markerstrokecolor=:black)
yticks!([4, 10, 20, 40, 60, 90], ["4", "10", "20", "40", "60", "90"])
h8 = display(plot!(sizes,Scale * modeloutput[:, 3], yaxis=:log2, label="11% Simulation",xlims=(0.25, 5.5), linewidth=2, color=:black))

h4 = plot(contrast11[!, 1], [4.97, 5.15, 7, 10.6, 16], ribbon=error_22, fillalpha=0.4, grid=false, legend=false, title="22%", xlims=(0.25, 5.5), ylabel="Phase-shift (deg)", titlefontsize=14, linewidth=1.5, yscale=:log2, markershape =:hexagon, markersize=7,yguidefontsize=13,xtickfontsize= 12, ytickfontsize=12, color=:steelblue2, markercolor=:steelblue2, markerstrokecolor=:black)
yticks!([3, 10, 20, 40, 60, 90], ["3", "10", "20", "40", "60", "90"])
h9 = display(plot!(sizes, Scale *modeloutput[:, 4], yaxis=:log2, label="22% Simulation", linewidth=2, xlims=(0.25, 5.5), color=:black))

h5 = plot(contrast46[!, 1], contrast46[!, 2], ribbon=error_46, fillalpha=0.4, grid=false, legend=false, title="46%",titlefontsize=14, xlims=(0.25, 5.5), linewidth=1.5, yscale=:log2, markershape =:utriangle, markersize=7 ,xtickfontsize= 12, ytickfontsize=12, color=:orange, markercolor=:orange, markerstrokecolor=:black)
yticks!([4, 10, 20, 40, 60, 90], ["4", "10", "20", "40", "60", "90"])
h10 = display(plot!(sizes, Scale *modeloutput[:, 5], yaxis=:log2, xlabel= "Gabor patch width (deg)", label="46% Simulation", xlims=(0.25, 5.5), color=:black, linewidth=2, xguidefontsize=16))

h61 = plot(contrast92[!, 1], contrast92[!, 2], ribbon=error_92, fillalpha=0.4, grid=false, legend=false, title="92%", xlims=(0.25, 5.5), titlefontsize=14, linewidth=1.5, yscale=:log2, markershape =:square, markersize=7,yguidefontsize=16, xtickfontsize= 12, ytickfontsize=12, color=:grey, markercolor=:grey, markerstrokecolor=:black)
yticks!([4, 10, 20, 40, 60, 90], ["4", "10", "20", "40", "60", "90"])
sizes1 = [0.63, 0.65, 0.67, 0.8, 0.9, 1.1, 1.3, 2.64, 4, 5]
th = [0.0625, 0.05024, 0.0464, 0.0424, 0.0398, 0.0379, 0.0366, 0.0484, 0.0743, 0.1129]
h11 = display(plot!(sizes,Scale * modeloutput[:, 6], yaxis=:log2,  label="92% Simulation", color=:black, linewidth=2, xlims=(0.25, 5.5)))

#fig1 = plot(h1, h2, h3, h4, h5, h61)
fig1 = plot(h1, h61)

savefig(fig1, "/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Adative_CS/Plots/ManyPlotsNoise")

# plot our model
display(plot(h1, h61))



# # End multiline comments
# for plots visit = http://julia.cookbook.tips/doku.php?id=plotattributes#plot_title
