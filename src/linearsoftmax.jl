import Distributions: Categorical
import Flux: softmax

struct LinearSoftmax <: AbstractPolicy end

export LinearSoftmax

function (π::LinearSoftmax)(θ, s)
    p = softmax(θ' * s)
    return Categorical(p)
end

function logpdf(π::LinearSoftmax, θ, s, a)
    p = softmax(θ' * s)
    return log(p[a])
end

function grad_logpdf!(ψ, π::LinearSoftmax, θ, s, a)
    p = -softmax(θ' * s)
    logpa = log(-p[a])
    p[a] += 1.0
    G = ψ'
    for i in 1:length(p)
        @. G[i, :] = s * p[i]
    end
    return logpa
end


function sample_with_trace!(ψ, action, π::LinearSoftmax, θ, s)
    p = softmax(θ' * s)
    action[1] = rand(Categorical(p))
    logp = log(p[action[1]])
    @. p *= -1
    p[action[1]] += 1.0
    G = ψ'
    for i in 1:length(p)
        @. G[i, :] = s * p[i]
    end
    return logp
end
