# DecisionMakingPolicies

<!--[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://DecisionMakingAI.github.io/DecisionMakingPolicies.jl/stable)-->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://DecisionMakingAI.github.io/DecisionMakingPolicies.jl/dev)
[![Build Status](https://github.com/DecisionMakingAI/DecisionMakingPolicies.jl/workflows/CI/badge.svg)](https://github.com/DecisionMakingAI/DecisionMakingPolicies.jl/actions)
[![Coverage](https://codecov.io/gh/DecisionMakingAI/DecisionMakingPolicies.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/DecisionMakingAI/DecisionMakingPolicies.jl)


The repository contains code useful for representing policies used in decision making problems. A policy, ``π``, is a function that returns distribution over actions. For example:

```julia
using Distributions
using DecisionMakingPolicies


num_actions = 4
π = StatelessSoftmax(num_actions)
d = π()             # returns a discrete distribution over 4 possible actions
θ = params(π)       # gets the policy parameters

a = rand(d)    # samples an action from the distribution d
logp = logpdf(d, a) # uses the Distributions.jl computation of the logpdf
```

Here ``π()`` returns a distribution over four possible actions. Since the policy takes not state or context as input we refer to it as a stateless (bandit) policy. The variable ``θ`` represent the parameters for a policy with four actions. The main functions for policies defined in this repo are: compute a distribution over actions (as shown above), compute the log probability of an action, compute the gradient of the log probability of the action, and sample and action and compute gradient of the log probability of the sampled action (useful for online algorithms). Below is an example use of these for a LinearSoftmax policy. 

```julia

num_features = 4
num_actions = 3
policy = LinearSoftmax(Float64, num_features, num_actions)   # note the Float64 is optional and the default but it will define the type for the weights.
x = randn(num_features)

d = policy(x)  # compute the conditional distributon over actions given the feature vector x
a = rand(d)  # sample the actions
logp = logpdf(d,a) # compute the log probability of a given x using Distributions.jl
logp = logpdf(policy, x, a)  # log probability of a given x
logps = logpdf(policy, x)  # compute the log probability for each action

logp, ψ = grad_logpdf(policy, x, a)  # compute the partial derivative of the log probability of a given a with respect to the policy parameters

a, logp, ψ = sample_with_trace(policy, x)  # sample an action, compute log probability of that action, and compute the partial derivative of with respect to the policy weights. This is faster than doing things individually. 
```

These functions are optimized, but still cause allocations which make them slow. To preallocate all the necessary arrays, there is a buffer struct for each policy type and a BufferedPolicy that will seemlessly handle the buffer. 

```julia
policy = LinearSoftmax(num_features, num_actions)
buffer = SoftmaxBuffer(policy)
bpolicy = BufferedPolicy(policy, buffer)

x = randn(num_features)
d = bpolicy(x)
a = 1 
logp, ψ = grad_logpdf(bpolicy, x, a)
a, logp, ψ = sample_with_trace(bpolicy, x)
```

Note that Buffers are not thread safe and are not supportive of autodifferentiation. Furthermore, they are only intended to be used for single samples not a batch of states and actions. This may be extended in the future (taking pullrequests).

The functions can also be used with autodifferentiation packages that support ChainRulesCore rules. For example:

```julia
using Zygote

policy = LinearSoftmax(num_features, num_actions)
x = rand(num_features)
a = 1
g = gradient(p->logpdf(p(x), a), policy)[1]  # gradient with respect to the struct policy
println(g.θ)  # partial derivative with respect to policy parameters
println(g[1]) # same as above
# Going through the Distributions.jl interface is slow to compute the gradients. The faster was is:
g = gradient(p->logpdf(p, x, a), policy)[1]

# The Flux way
import Flux
ps = Flux.params(params(policy)...)  # need to wrap parameters in a Flux object
g = gradient(()->logpdf(policy, x, a), ps)  # gradient with respect to the policy parameters. 
# Note that here we need to use logpdf(policy, x, a) not logpdf(policy(x), a) 
println(g[ps[1]])  # derivative with respect to policy parameters
```

We can also `seemlessly' use basis functions and neural networks. Examples below

```julia
using DecisionMakingUtils
using BenchmarkEnvironments
import Flux: Chain

env = cartpole_finitetime()

# construct a Fourier basis function 
function make_linearbasis(env, order)
    X = env.X # observation ranges
    n = size(X,1)
    
    nrm = ZeroOneNormalization(X)
    nbuff = zeros(n)
    nrm = BufferedFunction(nrm, nbuff)  # optional preallocations of the normalization function (do not use buffers unless you want single sample)
    fb = FourierBasis(n, order, order, false)  # create Fourier basis with given order
    num_features = length(fb)
    fbuff = FourierBasisBuffer(fb)
    fb = BufferedFunction(fb, fbuff) # optional preallocation of the basis function
    ϕ = Chain(nrm, fb)  # basis function can be any function. 
    return ϕ, num_features
end

ϕ, num_features = make_linearbasis(env, 3)
num_actions = length(env.A)
plin = LinearSoftmax(num_features, num_actions)
pbuff = SoftmaxBuffer(plin)
plin = BufferedPolicy(plin, pbuff)
policy = LinearPolicyWithBasis(plin, ϕ)

s,x = env.d0()
a, logp, ψ = sample_with_trace(policy, x)
```

For neural networks you can do the following:
```julia
import Flux

function nn_policy(env, num_layers, width, addnoise=false, normmode=:none)
    X = env.X 
    n = size(X,1)
    num_actions = length(env.A)
    
    # create normalizer
    if normmode == :none
        nrm = x->x
    elseif normmode == :zeroone
        nrm = ZeroOneNormalization(X)
    elseif normmode == :posneg
        nrm = PosNegNormalization(X)
    end
    act_fn = Flux.tanh  # can be whatever you want
    layers = []
    for i in 1:num_layers
        push!(layers,Flux.Dense(n, width, act_fn; bias=true))
        n = width
    end
    
    model = Flux.Chain(nrm, layers...)
    linpolicy = LinearSoftmax(n, num_actions)
    if addnoise  # optionally add noise to the last layer. I reccomend against this. 
        θ = params(linpolicy)[1]
        σ = √(2 / (n + num_actions))
        @. θ += randn() * σ
    end
    policy = LinearPolicyWithFluxBasis(linpolicy, model) # Flux Policies use the first layers as a trainable basis function. 
    
    return policy
end

env = cartpole_finitetime()
policy = nn_policy(env, 2, 32, false, :posneg)

s, x = env.d0()
d = policy(x)
a = rand(d)
ps = params(policy)  # LinearPolicyWithFluxBasis structs already use Flux.params object
g = gradient(()->logpdf(policy, x, a), ps)
```


Currently, we provide efficient implementations for:

- StatelessSoftmax
- StatelessNormal
- LinearSoftmax
- LinearNormal
