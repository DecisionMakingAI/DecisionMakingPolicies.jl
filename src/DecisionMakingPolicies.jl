module DecisionMakingPolicies

using DistributionsAD
using LinearAlgebra
import Distributions: logpdf, Categorical, Normal, MvNormal
using ChainRulesCore
import ChainRulesCore: rrule
import Zygote



abstract type AbstractPolicy end
abstract type AbstractStatelessPolicy <: AbstractPolicy end

export AbstractPolicy, AbstractStatelessPolicy
export StatelessSoftmax, LinearSoftmax, SoftmaxBuffer
export StatelessNormal, LinearNormal, NormalBuffer

struct ParameterizedPolicy{T} <: AbstractPolicy where {T}
    f::T
end

struct ParameterizedStatelessPolicy{T} <: AbstractStatelessPolicy where {T}
    f::T
end

struct LinearPolicyWithBasis{TP, TB} <: AbstractPolicy where {T, TB}
    π::TP
    ϕ::TB
end

function (π::ParameterizedPolicy)(s)
    return π.f(s)
end

function (π::ParameterizedStatelessPolicy)()
    return π.f()
end

function (π::LinearPolicyWithBasis)(s)
    feats = π.ϕ(s)
    return π.π(feats)    
end

function (π::LinearPolicyWithBasis)(s)
    feats = π.ϕ(s)
    return π.π(feats)    
end


export ParameterizedPolicy, ParameterizedStatelessPolicy
export LinearPolicyWithBasis

function sample_with_trace! end

export logpdf, logpdf!, grad_logpdf!, sample_with_trace!, initparams


function logpdf(π::AbstractPolicy, s, a)
    d = π(s)
    return logpdf(d, a)
end

function logpdf(π::LinearPolicyWithBasis, s, a)
    feats = π.ϕ(s)
    return logpdf(π.π, feats, a)
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

function grad_logpdf!(ψ, π::LinearPolicyWithBasis, s, a)
    feats = π.ϕ(s)
    logp = grad_logpdf!(ψ, π.π, feats, a)
    return logp
end

function grad_logpdf!(ψ, π::AbstractStatelessPolicy, a)
    logp, back = Zygote.pullback((π)->logpdf(π, a), π)
    ψ .= back(1)[1]
    return logp
end

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

function sample_with_trace!(ψ, action, π::LinearPolicyWithBasis, π, s)
    feats = π.ϕ(s)
    logp = sample_with_trace!(ψ, action, π.π, π, feats)
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


end
