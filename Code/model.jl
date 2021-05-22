# Simplified implementation of the model.
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
    b, center, amp = [80.7925, 0.2685, 6.5] # 80.7925, 0.2685, 6.5
    xb = 0.233#0.233
    if x < xb
        sig = 0.253  #0.253
    elseif c == 1 && x >= xb
        sig = amp/(1 + exp(-b *(x - center)))
    elseif c == 2 || c == 3 && x >= xb
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
# uncomment if noise is to be added.
"function Gnoise(t)
    n = 0.0
    if t > 0.0
        n = rand(Truncated(Normal(0, 1), -1, 1), 1)[1]
    end
    #display(n[1])
    return n
end"


# For simplicity, the output of the contrast normalization stage is built here.
kernel_fullsize = 3600
half = Int(kernel_fullsize/2)
I = zeros(kernel_fullsize)
arr_2_8 = Array{Float64}(undef, (kernel_fullsize, 5))
v_2_8 = [0.2266, 0.2266, 0.2267, 0.2270, 0.2272]
b28 = [ 0.2181, 0.2181, 0.2181, 0.2181, 0.2181]
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

# Determining the Gaussian RFs
fullsize = 3600
halfsize = fullsize/2
Gc = Array{Float64}(undef, fullsize)
Gs = Array{Float64}(undef, fullsize)
xi_ss = Array{Float64}(undef, (fullsize, 5)) # create a 400 x 3 array
y_ss = Array{Float64}(undef, 5) # create an array to store the ss of the output cell y.
x200_ss = Array{Float64}(undef, 5) # create an array to store the ss of the output cell y.
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


y = Array{Float64}(undef, (5, 6)) # array to store the steady state values
time_arr = Array{Float64}[] # array of any dims.
x_arr = Array{Float64}[] # array of any dims.

ratios = Array{Float64}(undef, (5, 6))
Inh_act = Array{Float64}(undef, (5, 6))
Exc_act = Array{Float64}(undef, (5, 6))
contraste = ["2.8%", "5.5%", "11%", "22%", "46%", "92%"]
for j=1:6 # for loop over the Data struct of contrast values
    fullsize = 3600
    # for loop to calculate Yex
    Yex = Array{Float64}(undef, (fullsize, 5))
    Yinh = Array{Float64}(undef, (fullsize, 5))

    for i = 1:fullsize
        # exitation
        Yex[i, 1] = RFc[i] * SigmoidC(Data[i, 1, j], j) # size = 0.63 deg
        Yex[i, 2] = RFc[i] * SigmoidC(Data[i, 2, j], j) # size = 1.31 deg
        Yex[i, 3] = RFc[i] * SigmoidC(Data[i, 3, j], j) # size = 2.64 deg
        Yex[i, 4] = RFc[i] * SigmoidC(Data[i, 4, j], j) # size = 4 deg
        Yex[i, 5] = RFc[i] * SigmoidC(Data[i, 5, j], j) # size = 5 deg
        # inhibition
        Yinh[i, 1] = RFs[i] * SigmoidS(Data[i, 1, j], j)
        Yinh[i, 2] = RFs[i] * SigmoidS(Data[i, 2, j], j)
        Yinh[i, 3] = RFs[i] * SigmoidS(Data[i, 3, j], j)
        Yinh[i, 4] = RFs[i] * SigmoidS(Data[i, 4, j], j)
        Yinh[i, 5] = RFs[i] * SigmoidS(Data[i, 5, j], j)
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
            du[1] = -A*u[1] + (B - u[1]) * YsumEx - (D + u[1]) * YsumInh #+ Gnoise(t)#rand(Truncated(Normal(0, 1), -1, 1), 1)[1]
        end

        function Gnoise(du, u, p, t)
            du[1] = 40.0
        end

        A, B, D, yEx, yInh = 1, 100, 50, Ex, Inh
        #display(Ex)
        p = [A, B, D, yEx, yInh] # Vector
        tspan = (0.0,4.0)
        prob = ODEProblem(f,[y0,],tspan,p) # Solving with no noise
        #prob = SDEProblem(f, Gnoise, [y0,],tspan,p) # Solving with noise

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
    #Y0 = (By * sumGc * SigmoidC(X0, 1) - Dy * sumGs * SigmoidS(X0, 1))/(A + sumGc * SigmoidC(X0, 1) + sumGs * SigmoidS(X0, 1))
    Y0 = (By * sumGc * 0.23 - Dy * sumGs * 0.253)/(A + sumGc * 0.23 + sumGs * 0.253)

    println(Y0)

    for k=1:5
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


# plotting the esum/isum ratio as a function of contrast
contrast = [2.8, 5.5, 11, 22, 46, 92]
plot(contrast, ratios[1,:], label="0.63 deg", xlabel="Contrast", ylabel="esum/isum", linewidth=2, shape=:circle, markersize=10)
plot!(contrast, ratios[2,:], label="1.31 deg", linewidth=2, shape=:hexagon, markersize=10)
plot!(contrast, ratios[3,:], label="2.64 deg", linewidth=2, shape=:utriangle, markersize=10)
plot!(contrast, ratios[4,:], label="4 deg", linewidth=2, shape=:square, markersize=10)
display(plot!(contrast, ratios[5,:], label="5 deg", linewidth=2, shape=:diamond, markersize=10))

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

# Plotting Exitation input and Inhibition input for each contrast as a funciton of size
norm_Exc_2_8 = Exc_act[:,1] ./ Exc_act[5, 6]
norm_Inh_2_8 = Inh_act[:,1] ./ Exc_act[5, 6]
norm_Exc_92 = Exc_act[:,6] ./ Exc_act[5, 6]
norm_Inh_92 = Inh_act[:,6] ./ Exc_act[5, 6]
plot(sizes, norm_Exc_2_8, label="Exc 2.8%", grid=false, xlabel="Size (deg)", ylabel="Normalized input activity", linewidth=2, shape=:circle, markersize=8)
plot!(sizes, norm_Inh_2_8, label="Inh 2.8%", linewidth=2, shape=:utriangle, markersize=8)
plot!(sizes, norm_Exc_92, label="Inh 92%", linewidth=2, shape=:square, markersize=8)
display(plot!(sizes, norm_Inh_92, label="Inh 92%", linewidth=2, shape=:diamond, markersize=8))


Threshold = 1 ./ y
display(Threshold)
save("/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/Noise/ModelOutput.jld", "xi_ss",  Threshold)

p1 = plot(sizes, Threshold[:, 1], yaxis=:log2, ylabel="Threshold", xlabel= "Size (deg)", title="Simulation", label="2.8%", linewidth=2.5, color=:black, shape=:circle, markersize=10)
p2 = plot!(sizes, Threshold[:, 2], yaxis=:log2, ylabel="Threshold", xlabel= "Size (deg)", label="5.5%", linewidth=2.5, color=:black, shape=:hexagon, markersize=10, markercolor=:white)
p3 = plot!(sizes, Threshold[:, 3], yaxis=:log2, ylabel="Threshold", xlabel= "Size (deg)", label="11%", linewidth=2.5, color=:black, shape=:utriangle, markersize=10, markercolor=:white)
p4 = plot!(sizes, Threshold[:, 4], yaxis=:log2, ylabel="Threshold", xlabel= "Size (deg)", label="22%", linewidth=2.5, color=:black, shape=:diamond, markersize=10, markercolor=:white)
p5 = plot!(sizes, Threshold[:, 5], yaxis=:log2, ylabel="Threshold", xlabel= "Size (deg)", label="46%", linewidth=2.5, color=:black, shape=:square, markersize=10, markercolor=:black)
p6 = display(plot!(sizes, Threshold[:, 6], yaxis=:log2, ylabel="Threshold", xlabel= "Size (deg)", label="92%", linewidth=2.5, color=:black, shape=:utriangle, markersize=10, markercolor=:black))

Scale = 5.1505547767110220489/Threshold[2,5] # fixed scale parameter
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



#q1 = plot(contrast2_8[!, 1], contrast2_8[!, 2], yerr = error_2_8, label="2.8%", color=:black, linewidth=2.5, yscale=:log2, markershape =:circle, markersize=10, markercolor=:black)
q1 = plot(contrast2_8[!, 1], contrast2_8[!, 2], fillalpha=0.4, title="Tadin's data", legend=true, label="2.8%", ylabel="Phase-shift (deg)", xlabel="Size (deg)", titlefontsize=14, grid=false, xlims=(0.25, 5.5), color=:salmon, linewidth=1.5, yscale=:log2, markershape =:circle, markersize=7,yguidefontsize=13, xtickfontsize= 12, ytickfontsize=12,  markercolor=:salmon, markerstrokecolor=:black)
yticks!([5, 10, 20, 40, 60, 90], ["5", "10", "20", "40", "60", "90"])
# ribbon = error_2_8
#q2 = plot!(contrast5_5[!, 1], contrast5_5[!, 2], yerr=error_5_5, label="5.5%", xlabel="Size deg", title="Tadin's data", color=:black, linewidth=2.5, yscale=:log2, markershape =:circle, markersize=10, markercolor=:white)
q2 = plot!(contrast5_5[!, 1], contrast5_5[!, 2], fillalpha=0.4, legend=true, label="5.5%", titlefontsize=14, grid=false, xlims=(0.25, 5.5), linewidth=1.5, yscale=:log2, markershape =:circle, markersize=7,xtickfontsize= 12, ytickfontsize=12,  color=:violet, markercolor=:violet, markerstrokecolor=:black)
#q3 = plot!(contrast11[!, 1], contrast11[!, 2], yerr=error_11, label="11%", color=:black, linewidth=2.5, yscale=:log2, markershape =:utriangle, markersize=10, markercolor=:white)
q3 = plot!(contrast11[!, 1], contrast11[!, 2], fillalpha=0.4, grid=false, legend=true, label="11%", xlims=(0.25, 5.5), titlefontsize=14, linewidth=1.5, yscale=:log2, markershape =:hexagon, markersize=7,xtickfontsize= 12, ytickfontsize=12,  color=:green, markercolor=:green, markerstrokecolor=:black)
#q4 = plot!(contrast22[!, 1], contrast22[!, 2], yerr=error_22, label="22%", color=:black, linewidth=2.5, yscale=:log2, markershape =:hexagon, markersize=10, markercolor=:white)
q4 = plot!(contrast11[!, 1], [4.97, 5.15, 7, 10.6, 16], fillalpha=0.4, grid=false, legend=true, label="22%", xlims=(0.25, 5.5), titlefontsize=14, linewidth=1.5, yscale=:log2, markershape =:hexagon, markersize=7,yguidefontsize=13,xtickfontsize= 12, ytickfontsize=12, color=:steelblue2, markercolor=:steelblue2, markerstrokecolor=:black)
#q5 = plot!(contrast46[!, 1], contrast46[!, 2], yerr=error_46, label="46%", color=:black, linewidth=2.5, yscale=:log2, markershape =:utriangle, markersize=10, markercolor=:white)
q5 = plot!(contrast46[!, 1], contrast46[!, 2], fillalpha=0.4, grid=false, legend=true, label="46%",titlefontsize=14, xlims=(0.25, 5.5), linewidth=1.5, yscale=:log2, markershape =:utriangle, markersize=7 ,xtickfontsize= 12, ytickfontsize=12, color=:orange, markercolor=:orange, markerstrokecolor=:black)
q6 = plot!(contrast92[!, 1], contrast92[!, 2], fillalpha=0.4, grid=false, legend=true, label="92%", xlims=(0.25, 5.5), titlefontsize=14, linewidth=1.5, yscale=:log2, markershape =:square, markersize=7,yguidefontsize=16, xtickfontsize= 12, ytickfontsize=12, color=:grey, markercolor=:grey, markerstrokecolor=:black)

# display multiple plots
display(plot(q1, p1_1, layout=(1, 2)))

fig1 = plot(h1, h2, h3, h4, h5, h61)
savefig(fig1, "/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Adative_CS/Plots/ManyPlots")

# Figure 6 of Paper2
element_size = contrast2_8[!, 1][1:3]
paper1_low = [70.05, 24.67, 16.67]
paper1_high = [2.85, 7.08, 10.08]
err_paper1_low = [6.14, 1.88, 1.63]
err_paper1_high = [0.38, 2.26, 2.13]
fig2 = plot(element_size, paper1_low, yerr= err_paper1_low, ylabel="Phase shift Threshold (deg)", xlabel="Gabor patch width (deg)", label="2.8% Peñaloza", linewidth=2.5, yscale=:log2, markershape =:utriangle, markersize=8, linestyle=:dot, grid=false, yguidefontsize=16, xguidefontsize=16, xtickfontsize= 14, ytickfontsize=14, color=:salmon, markercolor=:salmon)
yticks!([5, 10, 20, 40, 60, 90], ["5", "10", "20", "40", "60", "90"])
plot!(element_size, contrast2_8[!, 2][1:3], yerr= error_2_8[1:3], label="2.8% Tadin", linewidth=2.5, yscale=:log2, markershape =:square, markersize=8,  linestyle=:dash, color=:salmon, markercolor=:salmon)
plot!(element_size, Scale *Threshold[:, 1][1:3], label="2.8% Model", linewidth=2.5, yscale=:log2, markershape =:circle, markersize=8, color=:salmon, markercolor=:salmon)
plot!(element_size, paper1_high, yerr= err_paper1_high, label="92% Peñaloza", color=:grey, linewidth=2.5, yscale=:log2, markershape =:utriangle, markersize=8, markercolor=:grey, linestyle=:dot)
plot!(element_size, contrast92[!, 2][1:3], yerr=error_92[1:3], label="92% Tadin", color=:grey, linewidth=2.5, yscale=:log2, markershape =:square, markersize=8, markercolor=:grey, linestyle=:dash)
display(plot!(element_size, Scale *Threshold[:, 6][1:3], label="92% Model", color=:grey, linewidth=2.5, yscale=:log2, markershape =:circle, markersize=8, markercolor=:grey))
savefig(fig2, "/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Adative_CS/Plots/Penaloza_Tadin")

# Chi-square Goodness of fit
X2_28 = chiqtest(Scale *Threshold[:, 1], contrast2_8[!, 2], error_2_8, 5)
X2_55 = chiqtest(Scale *Threshold[:, 2], contrast5_5[!, 2], error_5_5, 5)
X2_11 = chiqtest(Scale *Threshold[:, 3], contrast11[!, 2], error_11, 5)
X2_22 = chiqtest(Scale *Threshold[:, 4], contrast22[!, 2], error_22, 5)
X2_46 = chiqtest(Scale *Threshold[:, 5], contrast46[!, 2], error_46, 5)
X2_92 = chiqtest(Scale *Threshold[:, 6], contrast92[!, 2], error_92, 5)
#X2_92 = chiqtest(Scale *Threshold[2:end, 6], contrast92[2][2:end], error_92[2:end]) # removing the first data point
println(ccdf(Chisq(4), X2_28))
println(ccdf(Chisq(4), X2_55))
println(ccdf(Chisq(4), X2_11))
println(ccdf(Chisq(4), X2_22))
println(ccdf(Chisq(4), X2_46))
println(ccdf(Chisq(4), X2_92))

chi2cum = X2_28 + X2_55 + X2_11 + X2_22 + X2_46 + X2_92
println(" ")
println(ccdf(Chisq(22), chi2cum))

# R-2 Goodness-of-fit
SS_2_8 = SSqs(Scale *Threshold[:, 1], contrast2_8[!, 2])
SS_5_5 = SSqs(Scale *Threshold[:, 2], contrast5_5[!, 2])
SS_11 = SSqs(Scale *Threshold[:, 3], contrast11[!, 2])
SS_22 = SSqs(Scale *Threshold[:, 4], contrast22[!, 2])
SS_46 = SSqs(Scale *Threshold[:, 5], contrast46[!, 2])
SS_92 = SSqs(Scale *Threshold[:, 6], contrast92[!, 2])

R2 = 1 - (SS_2_8[2] + SS_5_5[2] + SS_11[2] + SS_22[2] + SS_46[2] + SS_92[2])/(SS_2_8[1] + SS_5_5[1] + SS_11[1] + SS_22[1] + SS_46[1] + SS_92[1])
println("R2 model: ", R2)
p = 8
n = 30
R2_adj = 1 - (1 - 0.96) * (n-1)/(n - p - 1)
println("R2_adj model: ", R2_adj)
# plot our model
display(plot(h1, h2, h3, h4, h5, h61))


# Chi-square Goodness of fit for the Gain Model
GainModel = DataFrame!(CSV.File("/Users/Boris/Documents/MATLAB/gainmodel.csv", datarow=1))

X2_28_gm = chiqtest(GainModel[!, 1], contrast2_8[!, 2], error_2_8, 5)
X2_55_gm = chiqtest(GainModel[!, 2], contrast5_5[!, 2], error_5_5, 5)
X2_11_gm = chiqtest(GainModel[!, 3], contrast11[!, 2], error_11, 5)
X2_22_gm = chiqtest(GainModel[!, 4], contrast22[!, 2], error_22, 5)
X2_46_gm = chiqtest(GainModel[!, 5], contrast46[!, 2], error_46, 5)
X2_92_gm = chiqtest(GainModel[!, 6], contrast92[!, 2], error_92, 5)
chi2cum_gm = X2_28_gm + X2_55_gm + X2_11_gm + X2_22_gm + X2_46_gm + X2_92_gm
println(" ")
println(ccdf(Chisq(21), chi2cum_gm))

# R-2 Goodness-of-fit Gain Model
SS_gm_2_8 = SSqs(GainModel[!, 1], contrast2_8[!, 2])
SS_gm_5_5 = SSqs(GainModel[!, 2], contrast5_5[!, 2])
SS_gm_11 = SSqs(GainModel[!, 3], contrast11[!, 2])
SS_gm_22 = SSqs(GainModel[!, 4], contrast22[!, 2])
SS_gm_46 = SSqs(GainModel[!, 5], contrast46[!, 2])
SS_gm_92 = SSqs(GainModel[!, 6], contrast92[!, 2])
R2_gm = 1 - (SS_gm_2_8[2] + SS_gm_5_5[2] + SS_gm_11[2] + SS_gm_22[2] + SS_gm_46[2] + SS_gm_92[2])/(SS_gm_2_8[1] + SS_gm_5_5[1] + SS_gm_11[1] + SS_gm_22[1] + SS_gm_46[1] + SS_gm_92[1])
println("R2 Gain model: ", R2_gm)
p = 9
n = 30
R2_gm_adj = 1 - (1 - 0.98) * (n-1)/(n - p - 1)
println("R2_adj model: ", R2_gm_adj)

# plot
h1 = plot(contrast2_8[!, 1], contrast2_8[!, 2], ribbon = error_2_8, fillalpha=0.4, title="2.8%", legend=false, titlefontsize=14, grid=false, xlims=(0.25, 5.5), color=:salmon, linewidth=1.5, yscale=:log2, markershape =:circle, markersize=7,yguidefontsize=13, xtickfontsize= 12, ytickfontsize=12,  markercolor=:salmon, markerstrokecolor=:black)
yticks!([5, 10, 20, 40, 60, 90], ["5", "10", "20", "40", "60", "90"])
h6 = plot!(sizes, GainModel[!, 1], yaxis=:log2, ylabel="Phase-shift (deg)",xlims=(0.25, 5.5), linewidth=2, color=:grey)

h2 = plot(contrast5_5[!, 1], contrast5_5[!, 2], ribbon=error_5_5, fillalpha=0.4, title="5.5%", legend=false, titlefontsize=14, grid=false, xlims=(0.25, 5.5), linewidth=1.5, yscale=:log2, markershape =:circle, markersize=7,xtickfontsize= 12, ytickfontsize=12,  color=:violet, markercolor=:violet, markerstrokecolor=:black)
yticks!([5, 10, 20, 40, 60, 90], ["5", "10", "20", "40", "60", "90"])
h7 = plot!(sizes, GainModel[!, 2], yaxis=:log2, xlims=(0.25, 5.5), linewidth=2, color=:grey)

h3 = plot(contrast11[!, 1], contrast11[!, 2], ribbon=error_11, fillalpha=0.4, grid=false, legend=false, xlims=(0.25, 5.5), title="11%", titlefontsize=14, linewidth=1.5, yscale=:log2, markershape =:hexagon, markersize=7,xtickfontsize= 12, ytickfontsize=12,  color=:green, markercolor=:green, markerstrokecolor=:black)
yticks!([4, 10, 20, 40, 60, 90], ["4", "10", "20", "40", "60", "90"])
h8 = plot!(sizes,GainModel[!, 3], yaxis=:log2, label="11% Simulation",xlims=(0.25, 5.5), linewidth=2, color=:grey)

h4 = plot(contrast11[!, 1], [4.97, 5.15, 7, 10.6, 16], ribbon=error_22, fillalpha=0.4, grid=false, legend=false, title="22%", xlims=(0.25, 5.5), ylabel="Phase-shift (deg)", titlefontsize=14, linewidth=1.5, yscale=:log2, markershape =:hexagon, markersize=7,yguidefontsize=13,xtickfontsize= 12, ytickfontsize=12, color=:steelblue2, markercolor=:steelblue2, markerstrokecolor=:black)
yticks!([3, 10, 20, 40, 60, 90], ["3", "10", "20", "40", "60", "90"])
h9 = plot!(sizes, GainModel[!, 4], yaxis=:log2, label="22% Simulation", linewidth=2, xlims=(0.25, 5.5), color=:grey)

h5 = plot(contrast46[!, 1], contrast46[!, 2], ribbon=error_46, fillalpha=0.4, grid=false, legend=false, title="46%",titlefontsize=14, xlims=(0.25, 5.5), linewidth=1.5, yscale=:log2, markershape =:utriangle, markersize=7 ,xtickfontsize= 12, ytickfontsize=12, color=:orange, markercolor=:orange, markerstrokecolor=:black)
yticks!([4, 10, 20, 40, 60, 90], ["4", "10", "20", "40", "60", "90"])
h10 = plot!(sizes, GainModel[!, 5], yaxis=:log2, xlabel= "Gabor patch width (deg)", label="46% Simulation", xlims=(0.25, 5.5), color=:grey, linewidth=2, xguidefontsize=16)

h61 = plot(contrast92[!, 1], contrast92[!, 2], ribbon=error_92, fillalpha=0.4, grid=false, legend=false, title="92%", xlims=(0.25, 5.5), titlefontsize=14, linewidth=1.5, yscale=:log2, markershape =:square, markersize=7,yguidefontsize=16, xtickfontsize= 12, ytickfontsize=12, color=:grey, markercolor=:grey, markerstrokecolor=:black)
yticks!([4, 10, 20, 40, 60, 90], ["4", "10", "20", "40", "60", "90"])
h11 = plot!(sizes,GainModel[!, 6], yaxis=:log2,  label="92% Simulation", color=:grey, linewidth=2, xlims=(0.25, 5.5))
#display(plot(h1, h2, h3, h4, h5, h61))


# Chi-square Goodness of fit for the Gain Model
SizeModel = DataFrame!(CSV.File("/Users/Boris/Documents/MATLAB/sizemodel.csv", datarow=1))

X2_28_sm = chiqtest(SizeModel[!, 1], contrast2_8[!, 2], error_2_8, 5)
X2_55_sm = chiqtest(SizeModel[!, 2], contrast5_5[!, 2], error_5_5, 5)
X2_11_sm = chiqtest(SizeModel[!, 3], contrast11[!, 2], error_11, 5)
X2_22_sm = chiqtest(SizeModel[!, 4], contrast22[!, 2], error_22, 5)
X2_46_sm = chiqtest(SizeModel[!, 5], contrast46[!, 2], error_46, 5)
X2_92_sm = chiqtest(SizeModel[!, 6], contrast92[!, 2], error_92, 5)
chi2cum_sm = X2_28_sm + X2_55_sm + X2_11_sm + X2_22_sm + X2_46_sm + X2_92_sm
println(" ")
println(ccdf(Chisq(21), chi2cum_sm))

# R-2 Goodness-of-fit Size Model
SS_sm_2_8 = SSqs(SizeModel[!, 1], contrast2_8[!, 2])
SS_sm_5_5 = SSqs(SizeModel[!, 2], contrast5_5[!, 2])
SS_sm_11 = SSqs(SizeModel[!, 3], contrast11[!, 2])
SS_sm_22 = SSqs(SizeModel[!, 4], contrast22[!, 2])
SS_sm_46 = SSqs(SizeModel[!, 5], contrast46[!, 2])
SS_sm_92 = SSqs(SizeModel[!, 6], contrast92[!, 2])
R2_sm = 1 - (SS_sm_2_8[2] + SS_sm_5_5[2] + SS_sm_11[2] + SS_sm_22[2] + SS_sm_46[2] + SS_sm_92[2])/(SS_sm_2_8[1] + SS_sm_5_5[1] + SS_sm_11[1] + SS_sm_22[1] + SS_sm_46[1] + SS_sm_92[1])
println("R2 Size model: ", R2_sm)
p = 10
n = 30
R2_sm_adj = 1 - (1 - 0.97) * (n-1)/(n - p - 1)
println("R2_adj model: ", R2_sm_adj)

# plot
h1 = plot(contrast2_8[!, 1], contrast2_8[!, 2], ribbon = error_2_8, fillalpha=0.4, title="2.8%", legend=false, titlefontsize=14, grid=false, xlims=(0.25, 5.5), color=:salmon, linewidth=1.5, yscale=:log2, markershape =:circle, markersize=7,yguidefontsize=13, xtickfontsize= 12, ytickfontsize=12,  markercolor=:salmon, markerstrokecolor=:black)
yticks!([5, 10, 20, 40, 60, 90], ["5", "10", "20", "40", "60", "90"])
h6 = plot!(sizes, SizeModel[!, 1], yaxis=:log2, ylabel="Phase-shift (deg)",xlims=(0.25, 5.5), linewidth=2, color=:brown)
h6 = plot!(sizes, GainModel[!, 1], yaxis=:log2, ylabel="Phase-shift (deg)",xlims=(0.25, 5.5), linewidth=2, color=:grey)
h6 = plot!(sizes, Scale *Threshold[:, 1], yaxis=:log2, ylabel="Phase-shift (deg)",xlims=(0.25, 5.5), linewidth=2, color=:black)

h2 = plot(contrast5_5[!, 1], contrast5_5[!, 2], ribbon=error_5_5, fillalpha=0.4, title="5.5%", legend=false, titlefontsize=14, grid=false, xlims=(0.25, 5.5), linewidth=1.5, yscale=:log2, markershape =:circle, markersize=7,xtickfontsize= 12, ytickfontsize=12,  color=:violet, markercolor=:violet, markerstrokecolor=:black)
yticks!([5, 10, 20, 40, 60, 90], ["5", "10", "20", "40", "60", "90"])
h7 = plot!(sizes, SizeModel[!, 2], yaxis=:log2, xlims=(0.25, 5.5), linewidth=2, color=:brown)
h7 = plot!(sizes, GainModel[!, 2], yaxis=:log2, xlims=(0.25, 5.5), linewidth=2, color=:grey)
h7 = plot!(sizes, Scale *Threshold[:, 2], yaxis=:log2,xlims=(0.25, 5.5), linewidth=2, color=:black)

h3 = plot(contrast11[!, 1], contrast11[!, 2], ribbon=error_11, fillalpha=0.4, grid=false, legend=false, xlims=(0.25, 5.5), title="11%", titlefontsize=14, linewidth=1.5, yscale=:log2, markershape =:hexagon, markersize=7,xtickfontsize= 12, ytickfontsize=12,  color=:green, markercolor=:green, markerstrokecolor=:black)
yticks!([4, 10, 20, 40, 60, 90], ["4", "10", "20", "40", "60", "90"])
h8 = plot!(sizes,SizeModel[!, 3], yaxis=:log2, label="11% Simulation",xlims=(0.25, 5.5), linewidth=2, color=:brown)
h8 = plot!(sizes,GainModel[!, 3], yaxis=:log2, label="11% Simulation",xlims=(0.25, 5.5), linewidth=2, color=:grey)
h8 = plot!(sizes, Scale *Threshold[:, 3], yaxis=:log2,xlims=(0.25, 5.5), linewidth=2, color=:black)

h4 = plot(contrast11[!, 1], [4.97, 5.15, 7, 10.6, 16], ribbon=error_22, fillalpha=0.4, grid=false, legend=false, title="22%", xlims=(0.25, 5.5), ylabel="Phase-shift (deg)", titlefontsize=14, linewidth=1.5, yscale=:log2, markershape =:hexagon, markersize=7,yguidefontsize=13,xtickfontsize= 12, ytickfontsize=12, color=:steelblue2, markercolor=:steelblue2, markerstrokecolor=:black)
yticks!([3, 10, 20, 40, 60, 90], ["3", "10", "20", "40", "60", "90"])
h9 = plot!(sizes, SizeModel[!, 4], yaxis=:log2, label="22% Simulation", linewidth=2, xlims=(0.25, 5.5), color=:brown)
h9 = plot!(sizes, GainModel[!, 4], yaxis=:log2, label="22% Simulation", linewidth=2, xlims=(0.25, 5.5), color=:grey)
h9 = plot!(sizes, Scale *Threshold[:, 4], yaxis=:log2,xlims=(0.25, 5.5), linewidth=2, color=:black)

h5 = plot(contrast46[!, 1], contrast46[!, 2], ribbon=error_46, fillalpha=0.4, grid=false, legend=false, title="46%",titlefontsize=14, xlims=(0.25, 5.5), linewidth=1.5, yscale=:log2, markershape =:utriangle, markersize=7 ,xtickfontsize= 12, ytickfontsize=12, color=:orange, markercolor=:orange, markerstrokecolor=:black)
yticks!([4, 10, 20, 40, 60, 90], ["4", "10", "20", "40", "60", "90"])
h10 = plot!(sizes, SizeModel[!, 5], yaxis=:log2, xlabel= "Gabor patch width (deg)", label="46% Simulation", xlims=(0.25, 5.5), color=:brown, linewidth=2, xguidefontsize=16)
h10 = plot!(sizes, GainModel[!, 5], yaxis=:log2, xlabel= "Gabor patch width (deg)", label="46% Simulation", xlims=(0.25, 5.5), color=:grey, linewidth=2, xguidefontsize=16)
h10 = plot!(sizes, Scale *Threshold[:, 5], yaxis=:log2,xlims=(0.25, 5.5), linewidth=2, color=:black)

h61 = plot(contrast92[!, 1], contrast92[!, 2], ribbon=error_92, fillalpha=0.4, grid=false, legend=false, title="92%", xlims=(0.25, 5.5), titlefontsize=14, linewidth=1.5, yscale=:log2, markershape =:square, markersize=7,yguidefontsize=16, xtickfontsize= 12, ytickfontsize=12, color=:grey, markercolor=:grey, markerstrokecolor=:black)
yticks!([4, 10, 20, 40, 60, 90], ["4", "10", "20", "40", "60", "90"])
h11 = plot!(sizes,SizeModel[!, 6], yaxis=:log2,  label="92% Simulation", color=:brown, linewidth=2, xlims=(0.25, 5.5))
h11 = plot!(sizes,GainModel[!, 6], yaxis=:log2,  label="92% Simulation", color=:grey, linewidth=2, xlims=(0.25, 5.5))
h11 = plot!(sizes, Scale *Threshold[:, 6], yaxis=:log2,xlims=(0.25, 5.5), linewidth=2, color=:black)

mcomparison = plot(h1, h2, h3, h4, h5, h61)
display(mcomparison)
savefig(mcomparison, "/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/mcomp")


# # Chi-square Goodness of fit Penaloza data
X2_28_penaloza = chiqtest(Scale *Threshold[1:3, 1], paper1_low, err_paper1_low, 6)
X2_92_penaloza = chiqtest(Scale *Threshold[1:3, 6], paper1_high, err_paper1_high, 6)
println(" ")
println(ccdf(Chisq(3), X2_28_penaloza))
println(ccdf(Chisq(3), X2_92_penaloza))
println(" ")
println(ccdf(Chisq(6), X2_28_penaloza + X2_92_penaloza))

MDtask = 2.0412
MStask = 0.9097
println(ccdf(Chisq(10), MDtask+MStask))

contrast = [2.8, 5.5, 11, 22, 46, 92]
model1 = Scale*Threshold[1,:]
model2 = [67.96, 14.05, 9.43, 10.15, 10.33, 10.34]
plot(contrast, model1,  fillalpha=0.4, grid=false, legend=false,  label="Model 1",titlefontsize=14, xlims=(0., 100), linewidth=1.5, yscale=:log2, markershape =:utriangle, markersize=7 ,xtickfontsize= 12, ytickfontsize=12, color=:orange, markercolor=:orange, markerstrokecolor=:black)
yticks!([4, 10, 20, 40, 60, 90], ["4", "10", "20", "40", "60", "90"])
display(plot!(contrast, model2, yaxis=:log2, xlabel= "Contrast (%)", label="Model 2", xlims=(0., 100), color=:black, linewidth=2, xguidefontsize=16, markershape =:square))


# Gaussian input plot
fig = plot(contrast2_8[!, 1], contrast2_8[!, 2], ribbon = error_2_8, fillalpha=0.4, legend=true, label="data 2.8%", titlefontsize=14, grid=false, xlims=(0.25, 5.5), color=:salmon, linewidth=1.5, yscale=:log2,yguidefontsize=16, xguidefontsize=16, xtickfontsize= 12, ytickfontsize=12)
yticks!([5, 10, 20, 40, 60, 90], ["5", "10", "20", "40", "60", "90"])
plot!(contrast92[!, 1], contrast92[!, 2], ribbon=error_92, fillalpha=0.4, grid=false, legend=true, label="data 92%", xlims=(0.25, 5.5), titlefontsize=14, linewidth=1.5, yscale=:log2, yguidefontsize=16, xtickfontsize= 12, ytickfontsize=12, color=:grey)
plot!(sizes, Scale *Threshold[:, 1], yaxis=:log2, ylabel="Phase-shift (deg)", legend=true, xlabel="Gabor patch width (deg)", xlims=(0.25, 5.5), linewidth=2, color=:black, label="")
plot!(sizes,Scale * Threshold[:, 6], yaxis=:log2,  label="Rectangular pattern", color=:black, linewidth=2, xlims=(0.25, 5.5))
scatter!([0.63, 2.64, 5], Scale *[1/3.63, 1/16.7, 1/18.43], label="Gaussian pattern 2.8%", markershape =:utriangle, markersize=9, markercolor=:red)
display(scatter!([0.63, 2.64, 5], Scale *[1/42.8, 1/26.73, 1/11.45], label="Gaussian pattern 92%",markershape =:star4, markersize=9, markercolor=:blue))
savefig(fig, "/Users/Boris/Library/Mobile Documents/com~apple~CloudDocs/Boris_Documents/Second Paper/Modeling paper/Data/GaussianPattern")


# # End multiline comments
# for plots visit = http://julia.cookbook.tips/doku.php?id=plotattributes#plot_title
