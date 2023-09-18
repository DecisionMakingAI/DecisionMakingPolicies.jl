# DecisionMakingPolicies

<!--[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://DecisionMakingAI.github.io/DecisionMakingPolicies.jl/stable)-->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://DecisionMakingAI.github.io/DecisionMakingPolicies.jl/dev)
[![Build Status](https://github.com/DecisionMakingAI/DecisionMakingPolicies.jl/workflows/CI/badge.svg)](https://github.com/DecisionMakingAI/DecisionMakingPolicies.jl/actions)
[![Coverage](https://codecov.io/gh/DecisionMakingAI/DecisionMakingPolicies.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/DecisionMakingAI/DecisionMakingPolicies.jl)


The repository contains code useful for representing policies used in decision making problems. This library utilizes the [Lux.jl](https://github.com/LuxDL/Lux.jl) framework for managing the functional structure and parameters of the policy. A policy is a function that returns distribution over actions. For example:

```julia
using Distributions
using DecisionMakingPolicies
import Random
using ComponentArrays

num_actions = 4
policy = StatelessSoftmax(num_actions, init_weights=(rng, dims...)->zeros(Float64, dims...))
rng = Random.default_rng()

ps, st = setup(rng, policy)  # initializes policy parameters and state of the policy. 
ps = ps |> ComponentArray  # map the parameters to vector representation for easy manipulation. 

d, st = policy(ps, st)  # returns a distribution over actions along with new state of the policy.
a = rand(d)    # samples an action from the distribution d
logp = logpdf(d, a) # uses the Distributions.jl computation of the logpdf
```

Here ``policy(...)`` returns a distribution over four possible actions. Since the policy takes not state or context as input we refer to it as a stateless (bandit) policy. The main functions for policies defined in this repo are: 
- ``policy(...)`` compute a distribution over actions (as shown above),
- ``logpdf(policy, x, a, ps, st)`` compute the log probability of an action, 
- ``grad_logpdf(policy, x, a, ps, st)`` compute the gradient of the log probability of the action, and 
- ``sample_with_trace(policy, x, ps, st)`` sample an action and compute gradient of the log probability of the sampled action (useful for online algorithms). 

Below is an example use of these for a LinearSoftmax policy. 

```julia

num_features = 4
num_actions = 3
policy = LinearSoftmax(num_features, num_actions)   # default initialization is zeros of type Float32
ps, st = setup(rng, policy)  
x = randn(Float32, num_features)

d, st = policy(x, ps, st)  # compute the conditional distributon over actions given the feature vector x
a = rand(d)  # sample the actions
logp = logpdf(d,a) # compute the log probability of a given x using Distributions.jl
logp, st = logpdf(policy, x, a, ps, st)  # log probability of a given x
logps, st = logpdf(policy, x, a, ps, st)  # compute the log probability for each action

logp, ψ, st = grad_logpdf(policy, x, a, ps, st)  # compute the partial derivative of the log probability of a given a with respect to the policy parameters

a, logp, ψ, st = sample_with_trace(policy, x, ps, st)  # sample an action, compute log probability of that action, and compute the partial derivative of with respect to the policy weights. This is faster than doing things individually. 
```

These functions cause allocations so to make them faster we created a buffered implementation. To preallocate all the necessary arrays, the buffers are stored in the state of the policy. To use it simply at the buffered=true keyword in the constructor. 

```julia
policy = LinearSoftmax(num_features, num_actions, buffered=true)
ps, st = setup(rng, policy)

x = randn(num_features)
d = policy(x, ps, st)
a = 1 
logp, ψ, st = grad_logpdf(policy, x, a, ps, st)
a, logp, ψ, st = sample_with_trace(policy, x, ps, st)
```

The buffers are intended for fast linear policies working with a single samples. The buffers are ignored for batched input and are not always used for autodifferention. This may be extended in the future (taking pullrequests).

The policies can also be used with autodifferentiation packages that support ChainRulesCore rules. For example:

```julia
using Zygote

policy = LinearSoftmax(num_features, num_actions)
ps, st = setup(rng, policy)
x = rand(num_features)
a = 1
g = gradient(p->logpdf(policy(x, p, st)[1],a), ps)[1]  # gradient with respect to the policy parameters
# Going through the Distributions.jl interface is slow to compute the gradients. The faster was is:
g = gradient(p->logpdf(policy, x, a, ps, st)[1], ps)[1]
```

We can also `seemlessly' use basis functions and neural networks. Examples below

```julia
using DecisionMakingUtils
using BenchmarkEnvironments
import Lux

env = cartpole_finitetime()

# construct a Fourier basis function 
# note that this basis construction is an old style and will be migrated to the lux style soon. 
# However this shows a valid way to create an arbitrary basis function and incorporate it into the policy.
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
    ϕ(x) = fb(nrm(x))  # just some function that takes in an input x and outputs a vector of features
    return ϕ, num_features
end

ϕ, num_features = make_linearbasis(env, 3)
num_actions = length(env.A)
plinear = LinearSoftmax(num_features, num_actions, buffered=true)
policy = LinearPolicyWithBasis(ϕ, plinear)
ps, st = setup(rng, policy)

s,x = env.d0()
a, logp, ψ, st = sample_with_trace(policy, x, ps, st)
```

For neural networks you can do the following:
```julia

function nn_policy(env, num_layers, width)
    X = env.X 
    n = size(X,1)
    num_actions = length(env.A)
    
    act_fn = Lux.tanh  # can be whatever you want
    layers = []
    for i in 1:num_layers
        push!(layers,Lux.Dense(n => width, act_fn))
        n = width
    end
    
    model = Lux.Chain(layers...)
    linear_policy = LinearSoftmax(n, num_actions)
    policy = DeepPolicy(model, linear_policy)
    return policy
end

env = cartpole_finitetime()
policy = nn_policy(env, 2, 32)
ps, st = setup(rng, policy)

s, x = env.d0()
d, st = policy(x, ps, st)
a = rand(d)
g = gradient(p->logpdf(policy, x, a, p, st)[1], ps)[1]
logp, g2, st = grad_logpdf(policy, x, a, ps, st)
```

The functions ``logpdf`` and ``grad_logpdf`` also support batched data. For example:

```julia

input_dim = 4
action_dim = 3
policy = LinearNormal(input_dim, action_dim; init_weights=(rng, dims...)->zeros(Float64, dims...), init_std=(rng, dims...)->zeros(Float64, dims...))
ps, st = setup(rng, policy)

batch_size = 10
X = randn(Float64, input_dim, batch_size)  # last dim should always be batch_size
A = randn(Float64, action_dim, batch_size) 

logps, st = logpdf(policy, X, A, ps, st)  # compute log probability of each action in the batch
println(size(logps))  # (batch_size,)

logps, ψ, st = grad_logpdf(policy, X, A, ps, st)  # compute log probability of each action in the batch and the gradient of the sum log probabilities with respect to the policy parameters. 
println(size(logps))  # (batch_size,)
```

Currently, we provide implementations for:

- StatelessSoftmax
- StatelessNormal
- LinearSoftmax
- LinearNormal
- LinearPolicyWithBasis
- DeepPolicy
