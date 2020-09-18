struct StatelessSoftmax <: AbstractStatelessPolicy end
struct LinearSoftmax <: AbstractPolicy end

export StatelessSoftmax, LinearSoftmax


function initparams(π::StatelessSoftmax, ::Type{T}, action_dim::Int) where {T}
    N = action_dim
    θ = zeros(T, N)
    return θ
end

function initparams(π::StatelessSoftmax, action_dim::Int)
    return initparams(π, Float64, action_dim)
end

function (π::StatelessSoftmax)(θ)
    p = softmax(θ)
    return Categorical(p)
end

function (π::StatelessSoftmax)(θ, s)
    return π(θ)
end

function logpdf(π::StatelessSoftmax, θ, a)
    p = softmax(θ)
    return log(p[a])
end

function grad_logpdf!(ψ, π::StatelessSoftmax, θ, a)
    ψ .= -softmax(θ)
    logp = log(-ψ[a])
    ψ[a] += 1.0
    return logp
end


function sample_with_trace!(ψ, action, π::StatelessSoftmax, θ)
    ψ .= softmax(θ)
    a = rand(Categorical(ψ))
    action[1] = a

    logp = log(ψ[a])
    @. ψ *= -1.0
    ψ[a] += 1.0
    return logp
end


function (π::LinearSoftmax)(θ, s)
    p = softmax(θ' * s)
    return Categorical(p)
end

function initparams(π::LinearSoftmax, ::Type{T}, feature_dim::Int, action_dim::Int) where {T}
    M = feature_dim
    N = action_dim
    θ = zeros(T, (M,N))
    return θ
end

function initparams(π::LinearSoftmax, feature_dim::Int, action_dim::Int)
    return initparams(π, Float64, feature_dim, action_dim)
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
