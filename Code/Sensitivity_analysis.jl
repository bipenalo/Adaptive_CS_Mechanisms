# Code for the sensitivity analysis
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
function SigmoidC(x, c, center, b, amp)
    xb = 0.22
    if x < xb
        sig = 0.23 #0.23
        #sig = amp/(1 + exp(-b*(x - center  )))
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
function SigmoidS(x, c, center, b, amp)
    if center == 0.2685
        xb = 0.233
    else
        xb = 0.22#0.233
    end
    #xb = 0.22# 0.22
    if x < xb
        sig = 0.253  #0.253
        #sig = amp/(1 + exp(-b *(x - center)))
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

GABA = 0

# building the input stage after normalization for simplicity
kernel_fullsize = 3600
half = Int(kernel_fullsize/2)
I = zeros(kernel_fullsize)
arr_2_8 = Array{Float64}(undef, (kernel_fullsize, 5))
v_2_8 = [0.2266, 0.2266, 0.2267, 0.2270, 0.2272]
b28 = [ 0.2181, 0.2181, 0.218, 0.2179, 0.2176]
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


for n=1:5
    arr_2_8[:, n] = Inp_norm(I, n, v_2_8[n], b28[n], half)
    arr_5_5[:, n] = Inp_norm(I, n, v_5_5[n], b55[n], half)
    arr_11[:, n] = Inp_norm(I, n, v_11[n], b11[n], half)
    arr_22[:, n] = Inp_norm(I, n, v_22[n], b22[n], half)
    arr_46[:, n] = Inp_norm(I, n, v_46[n], b46[n], half)
    arr_92[:, n] = Inp_norm(I, n, v_92[n], b92[n], half)
end

# Store the matrixes in the respective containers
Data = Array{Float64}(undef, (size(arr_92)[1], size(arr_92)[2], 6)) # create a cell struct.. with 5 contrast
Data[:,:,1] = arr_2_8
Data[:,:,2] = arr_5_5
Data[:,:,3] = arr_11
Data[:,:,4] = arr_22
Data[:,:,5] = arr_46
Data[:,:,6] = arr_92

# C-S non-linearities
ac, bc = [75.25116013855349, 0.261]
as, bs = [80.25, 0.2685]
@. sigmoidC(x) = (6.8)/(1 + exp(-ac*(x - bc)))
@. sigmoidS(x) = (6.25)/(1 + exp(-as*(x - bs)))
xdata = range(0.0, stop=0.65, length=100)
sigC = sigmoidC(xdata) .+0.23
display(sigmoidS(0.2266))
#fig1_perc = plot([0.01, 0.22], [0.23, 0.23], legend=false, xguidefontsize=16, yguidefontsize=16,  xtickfontsize=14, ytickfontsize=14,xlims=(0.01, 0.15), color=:blue,grid=false, label="", legendfont=14, linewidth=4) #border=:none
fig1_perc = plot(xdata, sigC, xguidefontsize=16, yguidefontsize=16,  xlabel= "Contrast normalized activity", xtickfontsize=14, ytickfontsize=14,xlims=(0.15, 0.7), color=:blue,grid=false,  legend=false, legendfont=14, linewidth=4)
#vline!([0.22], label="2.8%", color=:grey, linewidth=2)
vline!([0.2266], label="2.8%", color=:grey, linewidth=2)
annotate!([(0.29,0.47,text("2.8%",16, :right, :top, :bold))])
vline!([0.6], label="92%", color=:black, linewidth=2)
annotate!([(0.66,0.47,text("92%",16, :right, :top, :bold))])

xdatas = range(0.0, stop=0.65, length=100)
sigS = sigmoidS(xdatas) .+ 0.25
#fig1_perc = plot!([0.01, 0.22], [0.253, 0.253], xguidefontsize=16, yguidefontsize=16,  xtickfontsize=14, ytickfontsize=14, color=:red,grid=false, legendfont=14, linewidth=4)
fig2_perc = plot!(xdatas, sigS,  color=:red,grid=false, linewidth=4)
#vline!([0.235], label="5.5%", color=:orange, linewidth=2)
display(vline!([0.6], label="92%", color=:black, linewidth=2))
savefig(fig1_perc, "/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/nonlinearities")

# Determining the Gaussian RFs
fullsize = 3600
halfsize = fullsize/2
Gc = Array{Float64}(undef, fullsize)
Gs = Array{Float64}(undef, fullsize)
xi_ss = Array{Float64}(undef, (fullsize, 5)) # create a 400 x 3 array
y_ss = Array{Float64}(undef, 5) # create an array to store the ss of the output cell y.
x200_ss = Array{Float64}(undef, 5) # create an array to store the ss of the output cell y.
RFc1 = Gaussian(Gc, 2/1, 100.0, halfsize, fullsize) # (2 and 67)
RFs1 = Gaussian(Gs, 1/1, 300.0, halfsize, fullsize) # (1 and 300)
Dog = RFc1 - RFs1
#plot(Dog, linewidth=8, grid=false)

plot(arr_2_8[:, 1], label="0.63 deg", xlab="space")
plot!(arr_2_8[:, 2], label="1.31 deg", xlab="space")
plot!(arr_2_8[:, 3], label="2.64 deg", xlab="space", linewidth=1)
plot!(arr_2_8[:, 4], label="4 deg", xlab="space")
display(plot!(arr_2_8[:, 5], label="5 deg", xlab="space", linewidth=1))
plot!((0.025*RFc1 .+0.218), label="RFc", xlab="space")
plot!(0.025*RFs1 .+0.218, label="RFs", xlab="space")
display(plot!((0.025*Dog.+0.218), label="DoG"))


y = Array{Float64}(undef, (5, 6)) # array to store the steady state values
# Array{Float64,N} where N(::UndefInitializer, ::Int64)
time_arr = Array{Float64}[] # array of any dims.
x_arr = Array{Float64}[] # array of any dims.

ratios = Array{Float64}(undef, (5, 6))
Inh_act = Array{Float64}(undef, (5, 6))
Exc_act = Array{Float64}(undef, (5, 6))
contraste = ["2.8%", "5.5%", "11%", "22%", "46%", "92%"]

Sensitivity = Array{Float64}(undef, (size(y)[1], size(y)[2], 7)) # create a cell struct with 6 containers representing each % change
param_change_ac = [0.253, 0.255, 0.258, 0.261, 0.263, 0.265, 0.267] # -3%, -2%, -1%, Sigmoid midpoint center non-linearity
param_change_as = [0.255, 0.263, 0.265, 0.2685, 0.271, 0.276, 0.279] # -5%, -3%,-1%,  Sigmoid midpoint surround non-linearity
sigmaC = [54, 60, 64, 67, 68, 74, 80] # -20%, -10%, -5%, 0%,
sigmaS = [240, 270, 285, 300, 315, 330, 360] # -20%, -10%, -5%, 0%,

slopeC = [64.2, 67.73, 71.49, 75.25, 79.01, 82.77, 86.5] # -15%, -10%, -5%, 0%,
slopeS = [68.21, 72.22, 76.24, 80.25, 84.26, 88.27, 92.28] # -15%, -10%, -5%, 0%,

AmpC = [5.95, 6.3, 6.65, 7, 7.35, 7.7, 8.05] # -15%, -10%, -5%, 0%,
AmpS = [5.52, 5.85, 6.18, 6.5, 6.82, 7.15, 7.47] # -15%, -10%, -5%, 0%,

param_change_B = [90, 95, 97.5, 100, 102.5, 105, 110] # Depolarization value B
tau = [0.00375, 0.05, 0.1, 1., 1.25, 1.5, 4]
bc, ampc = [75.251, 7.]
bs, amps = [80.251, 6.5]

for pv=1:7 # for loop over the % change in the sensitivity analysis
    RFc = Gaussian(Gc, 2/1, sigmaC[4], halfsize, fullsize) # (2 and 67)
    RFs = Gaussian(Gs, 1/1, sigmaS[4], halfsize, fullsize) # (1 and 300)
    for j=1:6 # for loop over the Data struct of contrast values
        fullsize = 3600
        # for loop to calculate Yex
        Yex = Array{Float64}(undef, (fullsize, 5))
        Yinh = Array{Float64}(undef, (fullsize, 5))

        for i = 1:fullsize
            # exitation
            Yex[i, 1] = RFc[i] * SigmoidC(Data[i, 1, j], j, param_change_ac[4], slopeC[4], AmpC[4]) # size = 0.63 deg
            Yex[i, 2] = RFc[i] * SigmoidC(Data[i, 2, j], j, param_change_ac[4], slopeC[4], AmpC[4]) # size = 1.31 deg
            Yex[i, 3] = RFc[i] * SigmoidC(Data[i, 3, j], j, param_change_ac[4], slopeC[4], AmpC[4]) # size = 2.64 deg
            Yex[i, 4] = RFc[i] * SigmoidC(Data[i, 4, j], j, param_change_ac[4], slopeC[4], AmpC[4]) # size = 4 deg
            Yex[i, 5] = RFc[i] * SigmoidC(Data[i, 5, j], j, param_change_ac[4], slopeC[4], AmpC[4]) # size = 5 deg
            # inhibition
            Yinh[i, 1] = RFs[i] * SigmoidS(Data[i, 1, j], j, param_change_as[4], slopeS[4], AmpS[4])
            Yinh[i, 2] = RFs[i] * SigmoidS(Data[i, 2, j], j, param_change_as[4], slopeS[4], AmpS[4])
            Yinh[i, 3] = RFs[i] * SigmoidS(Data[i, 3, j], j, param_change_as[4], slopeS[4], AmpS[4])
            Yinh[i, 4] = RFs[i] * SigmoidS(Data[i, 4, j], j, param_change_as[4], slopeS[4], AmpS[4])
            Yinh[i, 5] = RFs[i] * SigmoidS(Data[i, 5, j], j, param_change_as[4], slopeS[4], AmpS[4])
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
        function main(Ex, Inh, A, B, D , tau1, y0)

            function f(du, u, p, t)
                A, B, D, YsumEx, YsumInh, tau = p
                du[1] = tau*(-A*u[1] + (B - u[1]) * YsumEx - (D + u[1]) * YsumInh)
            end

            #A, B, D, yEx, yInh = 1, 100, 50, Ex, Inh
            #display(Ex)
            p = [A, B, D, Ex, Inh, tau1] # Vector
            tspan = (0.0,4.0)
            prob = ODEProblem(f,[y0,],tspan,p)
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
        B, A, By, Dy = 1, tau[pv], param_change_B[4], 50.0
        # initial conditions for neuron layer x
        I0 = L0*10 # first value after zero padding
        Ij = L0*35.837#8.837 35.837
        X0 = B*I0/(0.5 + I0 + Ij)
        println(X0)
        # initial conditions for neuron layer y
        Y0 = (By * sumGc * 0.23 - Dy * sumGs * 0.253)/(A + sumGc * 0.23 + sumGs * 0.253)
        println(Y0)
        #println(sumGc)

        #y = Array{Float64}(undef, (5, 1)) # array to store the steady state values
        for k=1:5
            out = main(Act_Ex[k],Act_Inh[k], tau[pv], param_change_B[4], 50, 1, Y0)
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

    # replace negative values to zeros
    y[y .< 0.0] .= 0.0


    # plotting the esum/isum ratio as a function of Size
    sizes = [0.63, 1.31, 2.64, 4, 5]
    ratios1 = Exc_act ./ Inh_act
    fig1 = plot(sizes, ratios1[:,1], label="2.8%", legendfont=font(12), legend=:topright, grid=false, xlabel="Size (deg)", ylabel="Ex/Inh", linewidth=4, shape=:circle, markersize=11, yguidefontsize=16,xtickfontsize= 14, ytickfontsize=14, color=:salmon, markercolor=:salmon)
    #plot!(sizes, ratios1[:,2], label="5.5%", linewidth=2, shape=:hexagon, markersize=10)
    #plot!(sizes, ratios1[:,3], label="11%", linewidth=2, shape=:utriangle, markersize=10)
    #plot!(sizes, ratios1[:,4], label="22%", linewidth=2, shape=:square, markersize=10)
    #plot!(sizes, ratios1[:,5], label="46%", linewidth=2, shape=:square, markersize=10)
    display(plot!(sizes, ratios1[:,6], label="92%", linewidth=4, xguidefontsize=16, shape=:square, markersize=11, color=:grey, markercolor=:grey))
    savefig(fig1, "/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Adative_CS/Plots/ratios_")

    # Plotting the Inhibitory contribution as a funciton of size
    Normalized_Inh_act = Inh_act ./ Inh_act[5,6]
    fig2 = plot(sizes, Normalized_Inh_act[:,1], label="2.8%", legendfont=font(12), grid=false, xlabel="Size (deg)", ylabel="Normalized Inhibitory Input", linewidth=3, shape=:circle, markersize=9, yguidefontsize=16,xguidefontsize=16, xtickfontsize= 14, ytickfontsize=14, color=:salmon, markercolor=:salmon)
    plot!(sizes, Normalized_Inh_act[:,2], label="5.5%", linewidth=3, shape=:utriangle, markersize=9, color=:violet, markercolor=:violet)
    plot!(sizes, Normalized_Inh_act[:,3], label="11%", linewidth=3, shape=:diamond, markersize=9, color=:green, markercolor=:green)
    plot!(sizes, Normalized_Inh_act[:,4], label="22%", linewidth=3, shape=:hexagon, markersize=9, color=:steelblue2, markercolor=:steelblue2)
    plot!(sizes, Normalized_Inh_act[:,5], label="46%", linewidth=3, shape=:utriangle, markersize=9,  color=:orange, markercolor=:orange)
    display(plot!(sizes, Normalized_Inh_act[:,6], label="92%", linewidth=3, shape=:square, markersize=9,  color=:grey, markercolor=:grey))
    savefig(fig2, "/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Adative_CS/Plots/NormInh_")

    Threshold = 1 ./ y
    display(Threshold)

    Threshold[Threshold .> 0.41] .= Inf # values larger than 90 deg are not plotted.


    Sensitivity[:,:,pv] = Threshold


    Scale = 5.1505547767110220489/Threshold[2,5]
    display(Scale)
    p1_1 = plot(sizes, Scale * Threshold[:, 1], fillalpha=0.4,  xlabel="Size (deg)", title="Model", legend=true, label="2.8%", titlefontsize=14, grid=false, xlims=(0.25, 5.5), color=:salmon, linewidth=1.5, yscale=:log2, markershape =:circle, markersize=7,yguidefontsize=13, xtickfontsize= 12, ytickfontsize=12,  markercolor=:salmon, markerstrokecolor=:black)
    yticks!([5, 10, 20, 40, 60, 90], ["5", "10", "20", "40", "60", "90"])
    p2_1 = plot!(sizes, Scale * Threshold[:, 2], fillalpha=0.4, legend=true, label="5.5%",titlefontsize=14, grid=false, xlims=(0.25, 5.5), linewidth=1.5, yscale=:log2, markershape =:circle, markersize=7,xtickfontsize= 12, ytickfontsize=12,  color=:violet, markercolor=:violet, markerstrokecolor=:black)
    p3_1 = plot!(sizes,  Scale * Threshold[:, 3], fillalpha=0.4, grid=false, legend=true, label="11%", xlims=(0.25, 5.5), titlefontsize=14, linewidth=1.5, yscale=:log2, markershape =:hexagon, markersize=7,xtickfontsize= 12, ytickfontsize=12,  color=:green, markercolor=:green, markerstrokecolor=:black)
    p4_1 = plot!(sizes,  Scale * Threshold[:, 4], fillalpha=0.4, grid=false, legend=true, label="22%", xlims=(0.25, 5.5), titlefontsize=14, linewidth=1.5, yscale=:log2, markershape =:hexagon, markersize=7,yguidefontsize=13,xtickfontsize= 12, ytickfontsize=12, color=:steelblue2, markercolor=:steelblue2, markerstrokecolor=:black)
    p5_1 = plot!(sizes,  Scale * Threshold[:, 5], fillalpha=0.4, grid=false, legend=true, label="46%", titlefontsize=14, xlims=(0.25, 5.5), linewidth=1.5, yscale=:log2, markershape =:utriangle, markersize=7 ,xtickfontsize= 12, ytickfontsize=12, color=:orange, markercolor=:orange, markerstrokecolor=:black)
    p6_1 = plot!(sizes,  Scale * Threshold[:, 6], fillalpha=0.4, grid=false, legend=true, label="92%", xlims=(0.25, 5.5), titlefontsize=14, linewidth=1.5, yscale=:log2, markershape =:square, markersize=7,yguidefontsize=16, xtickfontsize= 12, ytickfontsize=12, color=:grey, markercolor=:grey, markerstrokecolor=:black)


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
    h6 = display(plot!(sizes, Scale *Threshold[:, 1], yaxis=:log2, ylabel="Phase-shift (deg)",xlims=(0.25, 5.5), linewidth=2, color=:black))
    # shape=:circle, markersize=7, markercolor=:white
    h2 = plot(contrast5_5[!, 1], contrast5_5[!, 2], ribbon=error_5_5, fillalpha=0.4, title="5.5%", legend=false, titlefontsize=14, grid=false, xlims=(0.25, 5.5), linewidth=1.5, yscale=:log2, markershape =:circle, markersize=7,xtickfontsize= 12, ytickfontsize=12,  color=:violet, markercolor=:violet, markerstrokecolor=:black)
    yticks!([5, 10, 20, 40, 60, 90], ["5", "10", "20", "40", "60", "90"])
    h7 = display(plot!(sizes, Scale * Threshold[:, 2], yaxis=:log2, xlims=(0.25, 5.5), linewidth=2, color=:black))

    h3 = plot(contrast11[!, 1], contrast11[!, 2], ribbon=error_11, fillalpha=0.4, grid=false, legend=false, xlims=(0.25, 5.5), title="11%", titlefontsize=14, linewidth=1.5, yscale=:log2, markershape =:hexagon, markersize=7,xtickfontsize= 12, ytickfontsize=12,  color=:green, markercolor=:green, markerstrokecolor=:black)
    yticks!([4, 10, 20, 40, 60, 90], ["4", "10", "20", "40", "60", "90"])
    h8 = display(plot!(sizes,Scale * Threshold[:, 3], yaxis=:log2, label="11% Simulation",xlims=(0.25, 5.5), linewidth=2, color=:black))

    h4 = plot(contrast11[!, 1], [4.97, 5.15, 7, 10.6, 16], ribbon=error_22, fillalpha=0.4, grid=false, legend=false, title="22%", xlims=(0.25, 5.5), ylabel="Phase-shift (deg)", titlefontsize=14, linewidth=1.5, yscale=:log2, markershape =:hexagon, markersize=7,yguidefontsize=13,xtickfontsize= 12, ytickfontsize=12, color=:steelblue2, markercolor=:steelblue2, markerstrokecolor=:black)
    yticks!([3, 10, 20, 40, 60, 90], ["3", "10", "20", "40", "60", "90"])
    h9 = display(plot!(sizes, Scale *Threshold[:, 4], yaxis=:log2, label="22% Simulation", linewidth=2, xlims=(0.25, 5.5), color=:black))

    h5 = plot(contrast46[!, 1], contrast46[!, 2], ribbon=error_46, fillalpha=0.4, grid=false, legend=false, title="46%",titlefontsize=14, xlims=(0.25, 5.5), linewidth=1.5, yscale=:log2, markershape =:utriangle, markersize=7 ,xtickfontsize= 12, ytickfontsize=12, color=:orange, markercolor=:orange, markerstrokecolor=:black)
    yticks!([4, 10, 20, 40, 60, 90], ["4", "10", "20", "40", "60", "90"])
    h10 = display(plot!(sizes, Scale *Threshold[:, 5], yaxis=:log2, xlabel= "Gabor patch width (deg)", label="46% Simulation", xlims=(0.25, 5.5), color=:black, linewidth=2, xguidefontsize=16))

    h61 = plot(contrast92[!, 1], contrast92[!, 2], ribbon=error_92, fillalpha=0.4, grid=false, legend=false, title="92%", xlims=(0.25, 5.5), titlefontsize=14, linewidth=1.5, yscale=:log2, markershape =:square, markersize=7,yguidefontsize=16, xtickfontsize= 12, ytickfontsize=12, color=:grey, markercolor=:grey, markerstrokecolor=:black)
    yticks!([4, 10, 20, 40, 60, 90], ["4", "10", "20", "40", "60", "90"])
    sizes1 = [0.63, 0.65, 0.67, 0.8, 0.9, 1.1, 1.3, 2.64, 4, 5]
    th = [0.0625, 0.05024, 0.0464, 0.0424, 0.0398, 0.0379, 0.0366, 0.0484, 0.0743, 0.1129]
    h11 = display(plot!(sizes,Scale * Threshold[:, 6], yaxis=:log2,  label="92% Simulation", color=:black, linewidth=2, xlims=(0.25, 5.5)))
    #h12 = display(plot!(sizes1, Scale * th, yaxis=:log2,  label="92% Simulation", color=:black, linewidth=2, xlims=(0.25, 5.5)))


    fig1 = plot(h1, h2, h3, h4, h5, h61)
    savefig(fig1, "/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Adative_CS/Plots/ManyPlots")


    # plot our model
    display(plot(h1, h2, h3, h4, h5, h61))

end # end sensitivity loop

Scale = 218.4687883058878
sizes = [0.63, 1.31, 2.64, 4, 5]

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

#l = ["-5%", "-3%", "-1%", "1%", "3%", "5%"]
#l = ["-3%", "-2%", "-1%", "1%", "2%", "3%"]
#l= ["-15%", "-10%", "-5%", "5%", "10%", "15%"]
l= ["0.00375", "0.05", "0.1", "1.1", "1.5", "4"]


h1 = plot(contrast2_8[!, 1], contrast2_8[!, 2], ribbon = error_2_8, fillalpha=0.4, title="2.8%", legend=false, label="", titlefontsize=14, grid=false, xlims=(0.25, 5.5), color=:salmon, linewidth=1.5, yscale=:log2, markershape =:circle, markersize=7,yguidefontsize=13, xtickfontsize= 12, ytickfontsize=12,  markercolor=:salmon, markerstrokecolor=:black)
yticks!([5, 10, 20, 40, 60, 90], ["5", "10", "20", "40", "60", "90"])
plot!(sizes, Scale *Sensitivity[:, 1, 4], label="original", yaxis=:log2, ylabel="Phase-shift (deg)",xlims=(0.25, 5.5), linewidth=2, color=:black)
plot!(sizes, Scale *Sensitivity[:, 1, 1], yaxis=:log2, label= l[1], ylabel="Phase-shift (deg)",xlims=(0.25, 5.5), linewidth=2, linestyle=:dash, color=:red)
plot!(sizes, Scale *Sensitivity[:, 1, 2], yaxis=:log2, label= l[2], ylabel="Phase-shift (deg)",xlims=(0.25, 5.5), linewidth=2, linestyle=:dot, color=:red)
plot!(sizes, Scale *Sensitivity[:, 1, 3], yaxis=:log2, label=l[3], ylabel="Phase-shift (deg)",xlims=(0.25, 5.5), linewidth=2, linestyle=:dashdot, color=:red)
plot!(sizes, Scale *Sensitivity[:, 1, 5], yaxis=:log2, label=l[4], ylabel="Phase-shift (deg)",xlims=(0.25, 5.5), linewidth=2, linestyle=:dashdot, color=:blue)
plot!(sizes, Scale *Sensitivity[:, 1, 6], yaxis=:log2, label=l[5], ylabel="Phase-shift (deg)",xlims=(0.25, 5.5), linewidth=2, linestyle=:dot, color=:blue)
h6 = display(plot!(sizes, Scale *Sensitivity[:, 1, 7], yaxis=:log2, label=l[6], ylabel="Phase-shift (deg)",xlims=(0.25, 5.5), linewidth=2, linestyle=:dash, color=:blue))

h2 = plot(contrast5_5[!, 1], contrast5_5[!, 2], ribbon=error_5_5, fillalpha=0.4, title="5.5%", legend=false, titlefontsize=14, grid=false, xlims=(0.25, 5.5), linewidth=1.5, yscale=:log2, markershape =:circle, markersize=7,xtickfontsize= 12, ytickfontsize=12,  color=:violet, markercolor=:violet, markerstrokecolor=:black)
yticks!([5, 10, 20, 40, 60, 90], ["5", "10", "20", "40", "60", "90"])
plot!(sizes, Scale *Sensitivity[:, 2, 4], label="model", yaxis=:log2,xlims=(0.25, 5.5), linewidth=2, color=:black)
plot!(sizes, Scale *Sensitivity[:, 2, 1], yaxis=:log2, label= l[1],xlims=(0.25, 5.5), linewidth=2, linestyle=:dash,color=:red)
plot!(sizes, Scale *Sensitivity[:, 2, 2], yaxis=:log2, label= l[2],xlims=(0.25, 5.5), linewidth=2, linestyle=:dot,color=:red)
plot!(sizes, Scale *Sensitivity[:, 2, 3], yaxis=:log2, label= l[3],xlims=(0.25, 5.5), linewidth=2, linestyle=:dashdot,color=:red)
plot!(sizes, Scale *Sensitivity[:, 2, 5], yaxis=:log2, label=l[4],xlims=(0.25, 5.5), linewidth=2, linestyle=:dashdot,color=:blue)
plot!(sizes, Scale *Sensitivity[:, 2, 6], yaxis=:log2, label=l[5],xlims=(0.25, 5.5), linewidth=2, linestyle=:dot, color=:blue)
h7 = display(plot!(sizes, Scale *Sensitivity[:, 2, 7], yaxis=:log2, label=l[6],xlims=(0.25, 5.5), linewidth=2, linestyle=:dash, color=:blue))

h3 = plot(contrast11[!, 1], contrast11[!, 2], ribbon=error_11, fillalpha=0.4, grid=false, legend=true, label="", xlims=(0.25, 5.5), title="11%", titlefontsize=14, linewidth=1.5, yscale=:log2, markershape =:hexagon, markersize=7,xtickfontsize= 12, ytickfontsize=12,  color=:green, markercolor=:green, markerstrokecolor=:black)
yticks!([4, 10, 20, 40, 60, 90], ["4", "10", "20", "40", "60", "90"])
plot!(sizes, Scale *Sensitivity[:, 3, 4], label="model", legendfontsize=8, legend=:outertopright, yaxis=:log2,xlims=(0.25, 5.5), linewidth=2, color=:black)
plot!(sizes, Scale *Sensitivity[:, 3, 1], yaxis=:log2, label= l[1],xlims=(0.25, 5.5), linewidth=2, linestyle=:dash, color=:red)
plot!(sizes, Scale *Sensitivity[:, 3, 2], yaxis=:log2, label= l[2],xlims=(0.25, 5.5), linewidth=2, linestyle=:dot, color=:red)
plot!(sizes, Scale *Sensitivity[:, 3, 3], yaxis=:log2, label= l[3],xlims=(0.25, 5.5), linewidth=2, linestyle=:dashdot, color=:red)
plot!(sizes, Scale *Sensitivity[:, 3, 5], yaxis=:log2, label=l[4],xlims=(0.25, 5.5), linewidth=2, linestyle=:dashdot, color=:blue)
plot!(sizes, Scale *Sensitivity[:, 3, 6], yaxis=:log2, label=l[5],xlims=(0.25, 5.5), linestyle=:dot, linewidth=2, color=:blue)
h8 =display(plot!(sizes, Scale *Sensitivity[:, 3, 7], yaxis=:log2, label=l[6],xlims=(0.25, 5.5), linestyle=:dash, linewidth=2, color=:blue))


h4 = plot(contrast11[!, 1], [4.97, 5.15, 7, 10.6, 16], ribbon=error_22, fillalpha=0.4, grid=false, legend=false, title="22%", xlims=(0.25, 5.5), ylabel="Phase-shift (deg)", titlefontsize=14, linewidth=1.5, yscale=:log2, markershape =:hexagon, markersize=7,yguidefontsize=13,xtickfontsize= 12, ytickfontsize=12, color=:steelblue2, markercolor=:steelblue2, markerstrokecolor=:black)
yticks!([3, 10, 20, 40, 60, 90], ["3", "10", "20", "40", "60", "90"])
plot!(sizes, Scale *Sensitivity[:, 4, 4], label="model", yaxis=:log2, ylabel="Phase-shift (deg)",xlims=(0.25, 5.5), linewidth=2, color=:black)
plot!(sizes, Scale *Sensitivity[:, 4, 1], yaxis=:log2, label= "-5%", ylabel="Phase-shift (deg)",xlims=(0.25, 5.5), linewidth=2, linestyle=:dash,color=:red)
plot!(sizes, Scale *Sensitivity[:, 4, 2], yaxis=:log2, label= "-3%", ylabel="Phase-shift (deg)",xlims=(0.25, 5.5), linewidth=2, linestyle=:dot,color=:red)
plot!(sizes, Scale *Sensitivity[:, 4, 3], yaxis=:log2, label= "-1%", ylabel="Phase-shift (deg)",xlims=(0.25, 5.5), linewidth=2, linestyle=:dashdot,color=:red)
plot!(sizes, Scale *Sensitivity[:, 4, 5], yaxis=:log2, label="1%", ylabel="Phase-shift (deg)",xlims=(0.25, 5.5), linewidth=2, linestyle=:dashdot,color=:blue)
plot!(sizes, Scale *Sensitivity[:, 4, 6], yaxis=:log2, label="3%", ylabel="Phase-shift (deg)",xlims=(0.25, 5.5), linewidth=2, linestyle=:dot,color=:blue)
h9 = display(plot!(sizes, Scale *Sensitivity[:, 4, 7], yaxis=:log2, label="5%", ylabel="Phase-shift (deg)",xlims=(0.25, 5.5), linewidth=2, linestyle=:dash,color=:blue))


h5 = plot(contrast46[!, 1], contrast46[!, 2], ribbon=error_46, fillalpha=0.4, grid=false, legend=false, title="46%",titlefontsize=14, xlims=(0.25, 5.5), linewidth=1.5, yscale=:log2, markershape =:utriangle, markersize=7 ,xtickfontsize= 12, ytickfontsize=12, color=:orange, markercolor=:orange, markerstrokecolor=:black)
yticks!([4, 10, 20, 40, 60, 90], ["4", "10", "20", "40", "60", "90"])
plot!(sizes, Scale *Sensitivity[:, 5, 4], label="model", xlabel= "Gabor patch width (deg)", yaxis=:log2,xlims=(0.25, 5.5), linewidth=2, color=:black)
plot!(sizes, Scale *Sensitivity[:, 5, 1], yaxis=:log2, label= "-5%",xlims=(0.25, 5.5), linewidth=2, linestyle=:dash, color=:red)
plot!(sizes, Scale *Sensitivity[:, 5, 2], yaxis=:log2, label= "-3%",xlims=(0.25, 5.5), linewidth=2, linestyle=:dot, color=:red)
plot!(sizes, Scale *Sensitivity[:, 5, 3], yaxis=:log2, label= "-1%",xlims=(0.25, 5.5), linewidth=2, linestyle=:dashdot, color=:red)
plot!(sizes, Scale *Sensitivity[:, 5, 5], yaxis=:log2, label="1%",xlims=(0.25, 5.5), linewidth=2, linestyle=:dashdot, color=:blue)
plot!(sizes, Scale *Sensitivity[:, 5, 6], yaxis=:log2, label="3%",xlims=(0.25, 5.5), linewidth=2, linestyle=:dot, color=:blue)
h10 = display(plot!(sizes, Scale *Sensitivity[:, 5, 7], yaxis=:log2, label="5%",xlims=(0.25, 5.5), linewidth=2, linestyle=:dash, color=:blue))

h6 = plot(contrast92[!, 1], contrast92[!, 2], ribbon=error_92, fillalpha=0.4, grid=false, label="", legend=false, title="92%", xlims=(0.25, 5.5), titlefontsize=14, linewidth=1.5, yscale=:log2, markershape =:square, markersize=7,yguidefontsize=16, xtickfontsize= 12, ytickfontsize=12, color=:grey, markercolor=:grey, markerstrokecolor=:black)
yticks!([4, 10, 20, 40, 60, 90], ["4", "10", "20", "40", "60", "90"])
plot!(sizes, Scale *Sensitivity[:, 6, 4], label="original", yaxis=:log2,xlims=(0.25, 5.5), linewidth=2, color=:black)
plot!(sizes, Scale *Sensitivity[:, 6, 1], yaxis=:log2, label= "-5%",xlims=(0.25, 5.5), linewidth=2, linestyle=:dash, color=:red)
plot!(sizes, Scale *Sensitivity[:, 6, 2], yaxis=:log2, label= "-3%",xlims=(0.25, 5.5), linewidth=2, linestyle=:dot, color=:red)
plot!(sizes, Scale *Sensitivity[:, 6, 3], yaxis=:log2, label= "-1%",xlims=(0.25, 5.5), linewidth=2,linestyle=:dashdot, color=:red)
plot!(sizes, Scale *Sensitivity[:, 6, 5], yaxis=:log2, label="1%",xlims=(0.25, 5.5), linewidth=2, linestyle=:dashdot, color=:blue)
h11 = plot!(sizes, Scale *Sensitivity[:, 6, 6], yaxis=:log2, label="3%",xlims=(0.25, 5.5), linewidth=2, linestyle=:dot, color=:blue)
h12 = display(plot!(sizes, Scale *Sensitivity[:, 6, 7], yaxis=:log2, label="5%",xlims=(0.25, 5.5), linewidth=2, linestyle=:dash, color=:blue))



display(plot(h1, h2, h3, h4, h5, h6))
fig3= plot(h1, h2, h3, h4, h5, h6)
savefig(fig3, "/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Sensitivity_tau")

# # End multiline comments
# for plots visit = http://julia.cookbook.tips/doku.php?id=plotattributes#plot_title
