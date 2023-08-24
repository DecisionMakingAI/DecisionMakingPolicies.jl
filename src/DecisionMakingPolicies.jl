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

export LinearSoftmax#, StatelessSoftmax
# export StatelessNormal, LinearNormal, NormalBuffer
export LinearPolicyWithBasis
export LinearPolicyWithLuxBasis



include("softmax.jl")
# include("normal.jl")
# include("linearbasis.jl")
# include("flux.jl")
# include("linearfluxbasis.jl")
# include("nonlinearlux.jl")

end
