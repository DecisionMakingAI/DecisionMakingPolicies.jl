module DecisionMakingPolicies

using DistributionsAD
using LinearAlgebra
import Distributions: logpdf, Categorical, Normal, MvNormal, params
using ChainRulesCore
import ChainRulesCore: rrule, Tangent
import Zygote 
import Flux

export logpdf, grad_logpdf, sample_with_trace
export params, paramsvec, paramsfromvec!

export ParameterizedPolicy, ParameterizedStatelessPolicy
export AbstractPolicy, AbstractStatelessPolicy
export StatelessSoftmax, LinearSoftmax, SoftmaxBuffer
export StatelessNormal, LinearNormal, NormalBuffer
export LinearPolicyWithBasis
export LinearPolicyWithFluxBasis
export BufferedPolicy 

abstract type AbstractPolicy end
abstract type AbstractStatelessPolicy <: AbstractPolicy end


struct BufferedPolicy{TP,TB} <: AbstractPolicy where {TP,TB}
    π::TP
    buff::TB
end

function (π::BufferedPolicy)(s)
    return π.π(π.buff, s)
end

function logpdf(π::BufferedPolicy, s, a)
    return logpdf!(π.buff, π.π, s, a)
end

function grad_logpdf(π::BufferedPolicy, s, a)
    logp, ψ = grad_logpdf!(π.buff, π.π, s, a)
    return logp, ψ
end

function sample_with_trace(π::BufferedPolicy, s)
    action, logp, ψ =  sample_with_trace!(π.buff, π.π, s)
    return action, logp, ψ
end

function params(π::BufferedPolicy)
    return params(π.π)
end


include("softmax.jl")
include("normal.jl")
include("linearbasis.jl")
# include("flux.jl")
include("linearfluxbasis.jl")

end
