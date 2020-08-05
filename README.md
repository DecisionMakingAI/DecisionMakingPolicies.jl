# Policies

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://DecisionMakingAI.github.io/Policies.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://DecisionMakingAI.github.io/Policies.jl/dev)
[![Build Status](https://github.com/DecisionMakingAI/Policies.jl/workflows/CI/badge.svg)](https://github.com/DecisionMakingAI/Policies.jl/actions)
[![Coverage](https://codecov.io/gh/DecisionMakingAI/Policies.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/DecisionMakingAI/Policies.jl)


The repository contains code useful for representing policies used in decision making problems. A policy, ``π``, is a function that returns an action or distribution over actions for stochastic policies. For example:

```julia
using Distributions
import Flux: softmax


π = ParameterizedStatelessPolicy(θ->Categorical(softmax(θ)))
num_actions = 4
θ = zeros(num_actions)
d = π(θ)            # returns a discrete distribution over 4 possible actions

action = rand(d)    # samples an action from the distribution d
```

Here ``π(θ)`` returns a distribution over four possible actions. Since the policy takes not state or context as input we refer to it as a stateless (bandit) policy. The variable ``θ`` represent the parameters for a policy with four actions, but for this policy any vector could be used as policy parameters and return a distribution over N actions where N is the length of the vector. This is useful for making the policy a function structure and not dependent on a specific parameterization and allows for greater reuse.

We could have also a policy that takes a state as input and returns any distribution defined that implements the interface given in [Distributions.jl](https://github.com/JuliaStats/Distributions.jl). In the following example a beta distribution is defined using a linear transformation of the input.

```julia
function linear_beta(θ, s)
    params = θ' * s
    α, β = params[1], params[2]
    return Beta(α, β)
end

π = ParameterizedPolicy(linear_beta)

num_features = 10
θ = zeros(num_features,2)   # need two outputs of the linear function for α and β

state = rand(num_features)  # generate some random features as an example
d = π(θ, s)                 # returns a Beta distribution

action = rand(d)            # sample an action from the distribution.
```

The policies implemented thus are all for parameterized policies, meaning they take as input some parameters ``θ`` used in computing the distribution over actions.

In reinforcement learning it is common to need only a few functions aside from sample actions when interacting with a policy. The first is a way to compute the (natural) log probability of a given action, i.e., ``logpdf(π(θ,s),a)``. The second is to compute the partial derivative of the log probability of an action with respect to the policy parameters, i.e., ``∂/∂θ logpdf(π(θ,s),a)``. This can be accomplished using most autodiff packages, though we default to using Zygote.

```julia
import Zygote: gradient

d = π(θ,s)  # get the action distribution for some policy π
action = rand(d)  # sample some action
logp = logpdf(d, action)  # get the log density/probability using Distributions.jl


ψ = gradient(θ->logpdf(π(θ,s), action), θ)  # compute the gradient with respect to θ
```

The above code should work for most distributions and functions, however the computations are not always optimal. To allow more efficient implementations we define functions ``logpdf(π, θ, [s,] a)``, ``grad_logpdf!(ψ, π, θ, s, a)`` and ``sample_with_trace!(ψ, A, π, s)`` that can compute the same quantities as above, but can be specialized to each policy to decrease computation time and memory allocations.

- ``logpdf(π, θ, [s,] a)`` often just computes the density directly instead relying on the distributions implementation.
- ``grad_logpdf!(ψ, π, θ, s, a)`` computes the gradient with respect to ``θ`` and stores it in ``ψ``. It also returns ``logpdf(π, θ, s, a)`` since these are commonly needed together and easy to compute at the same time.
- ``sample_with_trace!(ψ, A, π, θ, s)`` samples an action storing it in A and computes the gradient storing it in ``ψ`` and returns ``logpdf(π, θ, s, a)``. This function makes online algorithms significantly more efficient.

Currently, we provide efficient implementations for:

- StatelessSoftmax
- StatelessNormal
- LinearSoftmax
- LinearNormal
