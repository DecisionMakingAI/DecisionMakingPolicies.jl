using Revise
import Zygote
import Flux
import ForwardDiff
using StaticArrays
using Distributions
using DistributionsAD
using BenchmarkTools



using Policies


function pol1(θ, x)
    p = Flux.softmax(θ' * x)
    return Categorical(p)
end

W = zeros(3,2)
σ = ones(2)
x = collect(Float64, 1:3)
p1 = ParameterizedPolicy(pol1)
A = zeros(2)
g = similar(W)
gσ = similar(σ)
logpdf(p1, W, x, 1)

p1 = StatelessSoftmax()
θ1 = initparams(p1, Float32, 10)
g1 = similar(θ1)
typeof(rand(p1(θ1)))
typeof(logpdf(p1(θ1),rand(p1(θ1))))
typeof(Zygote.gradient((θ)->logpdf(p1(θ),1), θ1)[1])
typeof(grad_logpdf!(g1, p1, θ1, 1))

p2 = LinearSoftmax()
θ2 = initparams(p2, Float32, 100, 5)
typeof(rand(p2(θ2, rand(100))))
typeof(logpdf(p2(θ2, rand(Float32, 100)),1))
typeof(logpdf(p2, θ2, rand(Float32, 100), 1))

p3 = StatelessNormal()
θ3 = initparams(p3, Float32, 7)

p4 = LinearNormal()
θ4 = initparams(p4, Float32, 25, 20)

tp = initparams(p4, Float32, 3, 2, Float32(1.0))

function tmp2(θ)
    N = round(Int,length(θ)/2)
    # μ, σ = θ[1:N], θ[N+1:end]
    if N > 1
        return MvNormal(θ[1:N], θ[N+1:end])
    else
        return Normal(θ[1], θ[2])
    end
end

W2 = Array([0.0, 1.0])
p2 = ParameterizedStatelessPolicy(tmp2)
g2 = similar(W2)
A2 = zeros(1)

function test_p2(g2, A2, p2, W2, x)
    for i in 1:100
        sample_with_trace!(g2, A2, p2, W2, x)
        # println(A2, g2)
    end
end


function test_p(g2, A2, p2, W2)
    for i in 1:1000
        sample_with_trace!(g2, A2, p2, W2)
        # println(A2, g2)
    end
end

function tmp(d, a)
    g = Zygote.gradient(Zygote.Params([mean(d), cov(d)])) do
        logpdf(d, a)
    end
    return g
end




function softmax(x)
    return exp.(x) / sum(exp.(x))
end


function softmax2(x)
    y = similar(x)
    maxx = maximum(x)
    @. y = x - maxx
    @. y = exp(y)
    y ./ sum(y)
    return y
end

# x = collect(Float64, 1:100)
# x2 = collect(Float64, 1:1000)
# sx = SVector{100,Float64}(1:100)
# sx2 = SVector{1000,Float64}(1:1000)
# numA = 100
# θ = zeros(Float64, (numA,length(x2)));
# θ2 = zeros(Float64, (length(x2), numA));

function logprob(θ, a)
    d = Categorical(Flux.softmax(θ))
    return logpdf(d, a)
end

function logprob2(θ, a)
    p = Flux.softmax(θ)
    return log(p[a])
end

function linear_softmax(θ, x)
    y = θ * x
    d = Flux.softmax(y)
    return d
end

# function linear_softmax2(θ, x)
#     y = vec(x' * θ)
#     # d = Categorical(Flux.softmax(y))
#     d = Flux.softmax(y)
#     return d
# end

function logprob_linear(θ, x, a)
    d = linear_softmax(θ,x)
    # return logpdf(d,a)
    return log(d[a])
end

function logprob_linear3(θ, x, a)
    # d = linear_softmax(θ,x)
    y = linear_softmax(θ, x)
    return log(y[a])
    # d = Categorical(linear_softmax(θ,x))
    # return logpdf(Categorical(d), a)
    # println(d)
    # println(a)
    # println(d.p)
    # @code_warntype(logpdf(d,a))
    # logpa = logpdf(d,a)
    # return logpa
    # return logpdf(d,a)
    # return log(probs(d)[a])

    # return log(d[a])
end

# @adjoint function sort(x)
#      p = sortperm(x)
#      x[p], x̄ -> (x̄[invperm(p)],)
# end

# Zygote.@adjoint function logpdf(d::DiscreteNonParametric{Int64,P,Base.OneTo{Int64},Ps} where Ps<:AbstractArray{P,1} where P<:Real, a::Int64)
#     ptmp = similar(d.p)
#     fill!(ptmp, 0.0)
#     ptmp[a] = 1.0
#     return logpdf(d,a), x̄->(nothing, x̄ .* (ptmp .- d.p), nothing)
# end

function grad_logprob!(G, θ, x, a)
    p = -Flux.softmax(θ * x)
    logpa = log(-p[a])
    p[a] += 1.0
    for i in 1:length(p)
        @. G[i, :] = x * p[i]
    end
    return logpa
end

# Zygote.@adjoint function logprob_linear(θ, x, a)
#     g = similar(θ)
#     # gx = similar(x)
#     p = -Flux.softmax(θ * x)
#     logpa = log(-p[a])
#     p[a] += 1.0
#     # fill!(gx, 0.0)
#     for i in 1:length(p)
#         @. g[i, :] = x * p[i]
#         # @. gx += p[i] * θ[i, :]
#     end
#
#
#     # return logpa, p̄ ->((@. g * p̄), (@. gx * p̄), (nothing))
#     return logpa, p̄ ->((@. g * p̄), (nothing), (nothing))
# end

# function logprob_linear2(θ, x, a)
#     d = linear_softmax2(θ,x)
#     # return logpdf(d,a)
#     return log(d[a])
# end

function gradlogprob_linear3(θ, x, a)
    g = Zygote.gradient((θ)->logprob_linear3(θ,x,a), θ)
end

function gradlogprob_linear(θ, x, a)
    g = Zygote.gradient((θ)->logprob_linear(θ,x,a), θ)
end

# function gradlogprob_linear2(θ, x, a)
#     g = Zygote.gradient((θ)->logprob_linear2(θ,x,a), θ)
# end

function fgradlogprob_linear(θ, x, a)
    # g = Zygote.gradient(θ) do p
    #     Zygote.forwarddiff(p) do p
    #         logprob_linear(p, x, a)
    #     end
    # end
    ForwardDiff.gradient((θ)->logprob_linear(θ, x, a), θ)
end

function fgradlogprob_linear2(θ, x, a)
    ForwardDiff.gradient((θ)->logprob_linear2(θ, x, a), θ)
    # g = Zygote.gradient(θ) do p
    #     Zygote.forwarddiff(p) do p
    #         logprob_linear2(p, x, a)
    #     end
    # end
end

function bench_linear(::Type{T}) where {T}
    for numA in [2,4,10,100,1000]
        for numX in [2,4,10,100,1000,10000]
            println("numA: $numA\t numX: $numX")
            x = collect(T, 1:numX) ./ numX;
            θ = zeros(T, (numA, numX))
            θ2 = zeros(T, (numX, numA))
            # @btime linear_softmax($θ, $x) # winner with two actions and second place until large feature space (>=1000)
            @btime linear_softmax($(θ2)', $x) # winner overall
            # @btime linear_softmax2($θ', $x) # always worse?
            # @btime linear_softmax2($θ2, $x) # second place for large action space and feature space
        end
    end
end

function bench_linearp(::Type{T}) where {T}
    for numX in [2,4,10,100,1000,10000]
        for numA in [2,4,10,100,1000]
            println("numA: $numA\t numX: $numX")
            x = collect(T, 1:numX) ./ numX;
            θ = zeros(T, (numA, numX))
            θ2 = zeros(T, (numX, numA))
            # @btime logprob_linear($θ, $x, $numA) # winner with two actions and second place until large feature space (>=1000)
            @btime logprob_linear3($(θ2)', $x, $numA) # winner overall
            # @btime logprob_linear2($θ', $x, $numA) # always worse?
            # @btime logprob_linear3($θ2, $x, $numA) # second place for large action space and feature space
        end
    end
end

function bench_lineargp(::Type{T}) where {T}
    for numX in [100,1000,10000]
        for numA in [2,4,10,100,1000]#[2,4,10,100,1000]
            println("numA: $numA\t numX: $numX")
            # x = collect(T, 1:numX) ./ numX;
            # x2 = collect(Float32, 1:numX) ./ numX;
            x3 = collect(Float64, 1:numX) ./ numX;
            # θ = zeros(T, (numA, numX))
            # θ2 = zeros(Float32, (numX, numA))
            # g = similar(θ)
            # g2 = similar(θ2)
            θ3 = zeros(Float64, (numX, numA))
            g3 = similar(θ3)
            # θ = zeros(SMatrix{numA,numX,T})
            # θ3 = zeros(SMatrix{numX,numA,T})
            # g = zeros(SMatrix{numA,numX,T})
            # g3 = zeros(SMatrix{numX,numA,T})
            # @btime gradlogprob_linear($θ, $x, $numA) # winner with two actions and second place until large feature space (>=1000)
            # @btime gradlogprob_linear($(θ2)', $x2, $numA) # winner overall
            # @btime grad_logprob!($g, $(θ), $x, $numA)
            # @btime grad_logprob!($(g2)', $(θ2)', $x2, $numA)
            # @btime grad_logprob!($(g3)', $(θ3)', $x3, $numA)
            # @btime gradlogprob_linear3($(θ2)', $x2, $numA) # winner overall
            @btime gradlogprob_linear3($(θ3)', $x3, $numA) # winner overall

            # @btime gradlogprob_linear2($θ2, $x, $numA) # second place for large action space and feature space

            # @btime fgradlogprob_linear($θ, $x, $numA) # winner with two actions and second place until large feature space (>=1000)
            # @btime fgradlogprob_linear($(θ2)', $x, $numA) # winner overall
            # @btime fgradlogprob_linear2($θ2, $x, $numA) # second place for large action space and feature space
        end
    end
end


function sample_action_wtrace!(G, θ, x)
    p = Flux.softmax(θ * x)
    a = rand(Categorical(p))
    logpa = log(p[a])
    @. p *= -1
    p[a] += 1.0
    for i in 1:length(p)
        @. G[i, :] = x * p[i]
    end
    return a, logpa
end

function sample_action_wtrace(θ, x)
    function f!(res, θ,x)
        # d = linear_softmax(θ, x)
        p = Flux.softmax(θ*x)

        # println("action inside: $a")
        Zygote.ignore() do
            d = Categorical(p)
            a = rand(d)
            res[1] = a
        end
        return log(p[res[1]])#logpdf(d, a)
    end
    res = Array{Int,1}([0])
    p, back = Zygote.pullback(θ->f!(res,θ,x), θ)
    # println("action stored: $(res[1])")
    action = res[1]
    return action, p, back(1);
end


function bench_linearsgp(::Type{T}) where {T}
    for numX in [100,1000,10000]
        for numA in [2,4,10,100,1000]#[2,4,10,100,1000]
            println("numA: $numA\t numX: $numX")

            x3 = collect(Float64, 1:numX) ./ numX;
            θ3 = zeros(Float64, (numX, numA))
            g3 = similar(θ3)
            @btime sample_action_wtrace!($(g3)', $(θ3)', $x3)
            @btime sample_action_wtrace($(θ3)', $x3)
        end
    end
end
