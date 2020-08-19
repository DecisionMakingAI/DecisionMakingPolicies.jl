module DecisionMakingPolicies

using DistributionsAD
import NNlib: softmax
import Distributions: logpdf, Categorical, Normal, MvNormal
import Zygote

abstract type AbstractPolicy end
abstract type AbstractStatelessPolicy <: AbstractPolicy end

export AbstractPolicy, AbstractStatelessPolicy


struct ParameterizedPolicy{T} <: AbstractPolicy where {T}
    f::T
end

struct ParameterizedStatelessPolicy{T} <: AbstractStatelessPolicy where {T}
    f::T
end

function (π::ParameterizedPolicy)(θ, s)
    return π.f(θ, s)
end

function (π::ParameterizedStatelessPolicy)(θ)
    return π.f(θ)
end


export ParameterizedPolicy, ParameterizedStatelessPolicy

function sample_with_trace end
function compatiable_features end
function init_params end

export logpdf, grad_logpdf!, sample_with_trace!, initparams


function logpdf(π::AbstractPolicy, θ, s, a)
    d = π(θ, s)
    return logpdf(d, a)
end

function logpdf(π::AbstractStatelessPolicy, θ, a)
    d = π(θ)
    return logpdf(d, a)
end

function grad_logpdf!(ψ, π::AbstractPolicy, θ, s, a)
    logp, back = Zygote.pullback((θ)->logpdf(π, θ, s, a), θ)
    ψ .= back(1)[1]
end

function grad_logpdf!(ψ, π::AbstractStatelessPolicy, θ, a)
    logp, back = Zygote.pullback((θ)->logpdf(π, θ, a), θ)
    ψ .= back(1)[1]
    return logp
end

function sample_with_trace!(ψ, action, π::AbstractPolicy, θ, s)
    function f!(res, π, θ, s)
        d = π(θ, s)

        Zygote.ignore() do
            a = rand(d)
            if length(a) == 1
                res[1] = a
            else
                res .= a
            end
        end
        if length(d) > 2
            return logpdf(d, res)
        else
            return logpdf(d, res[1])
        end
    end
    logp, back = Zygote.pullback(θ->f!(action, π, θ, s), θ)
    ψ .= back(1)[1]
    return logp
end

function sample_with_trace!(ψ, action, π::AbstractStatelessPolicy, θ)
    function f!(res, π, θ)
        d = π(θ)

        Zygote.ignore() do
            a = rand(d)
            if length(a) == 1
                res[1] = a
            else
                res .= a
            end
            # println(a, res)
        end
        if length(d) > 2
            return logpdf(d, res)
        else
            return logpdf(d, res[1])
        end
    end
    logp, back = Zygote.pullback(θ->f!(action, π, θ), θ)
    ψ .= back(1)[1]
    return logp
end

include("softmax.jl")
include("normal.jl")


end
