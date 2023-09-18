module DecisionMakingPolicies

using Reexport
import LuxCore
@reexport using LuxCore: setup

using LinearAlgebra
import Distributions: logpdf, Categorical, Normal, MvNormal
using ChainRulesCore
import ChainRulesCore: rrule
import Zygote 
import Random
using ComponentArrays

export logpdf, grad_logpdf, sample_with_trace

export LinearSoftmax, StatelessSoftmax
export StatelessNormal, LinearNormal
export LinearPolicyWithBasis
export DeepPolicy

LuxCore.zeros(rng::Random.AbstractRNG, size...) = LuxCore.zeros(size...)
zeros32(rng::Random.AbstractRNG, size...) = LuxCore.zeros(Float32, size...)

include("softmax_stateless.jl")
include("softmax_linear.jl")
include("normal_stateless.jl")
include("normal_linear.jl")
include("luxnetwork.jl")
include("linearbasis.jl")

end
