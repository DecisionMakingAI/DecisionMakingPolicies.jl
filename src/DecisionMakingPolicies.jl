module DecisionMakingPolicies

using DistributionsAD
using LinearAlgebra
import Distributions: logpdf, Categorical, Normal, MvNormal
using ChainRulesCore
import ChainRulesCore: rrule
import Zygote 
import Flux

export logpdf, logpdf!, grad_logpdf!, sample_with_trace!, params

export ParameterizedPolicy, ParameterizedStatelessPolicy
export AbstractPolicy, AbstractStatelessPolicy
export StatelessSoftmax, LinearSoftmax, SoftmaxBuffer
export StatelessNormal, LinearNormal, NormalBuffer
export LinearPolicyWithBasis, LinearWithBasisBuffer
export FluxPolicy, FluxBuffer

abstract type AbstractPolicy end
abstract type AbstractStatelessPolicy <: AbstractPolicy end


struct ParameterizedPolicy{T} <: AbstractPolicy where {T}
    f::T
end

struct ParameterizedStatelessPolicy{T} <: AbstractStatelessPolicy where {T}
    f::T
end

function (π::ParameterizedPolicy)(s)
    return π.f(s)
end

function (π::ParameterizedStatelessPolicy)()
    return π.f()
end


function logpdf(π::AbstractPolicy, s, a)
    d = π(s)
    return logpdf(d, a)
end

function logpdf(π::AbstractStatelessPolicy, a)
    d = π()
    return logpdf(d, a)
end

function grad_logpdf!(ψ, π::AbstractPolicy, s, a)
    logp, back = Zygote.pullback((π)->logpdf(π, s, a), π)
    ψ .= back(1)[1]
    return logp
end

function grad_logpdf!(ψ, π::AbstractStatelessPolicy, a)
    logp, back = Zygote.pullback((π)->logpdf(π, a), π)
    ψ .= back(1)[1]
    return logp
end

function sample_with_trace! end

function sample_with_trace!(ψ, action, π::AbstractPolicy, s)
    function f!(res, π, s)
        d = π(s)

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
    logp, back = Zygote.pullback(π->f!(action, π, s), π)
    ψ .= back(1)[1]
    return logp
end

function sample_with_trace!(ψ, action, π::AbstractStatelessPolicy)
    function f!(res, π)
        d = π()

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
    logp, back = Zygote.pullback(π->f!(action, π), π)
    ψ .= back(1)[1]
    return logp
end



include("softmax.jl")
include("normal.jl")
include("linearbasis.jl")
include("flux.jl")

end
