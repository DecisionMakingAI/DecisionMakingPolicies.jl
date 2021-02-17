struct StatelessSoftmax <: AbstractStatelessPolicy end
struct LinearSoftmax <: AbstractPolicy end

export StatelessSoftmax, LinearSoftmax

function softmax2!(out::Vector{T}, x::Vector{T}) where {T}
    @. out = x
    softmax2!(out)
end

function softmax2!(x::Vector{T}) where {T}
    maxx = maximum(x)
    x .-= maxx
    @. x = exp(x)
    x ./= sum(x)
    return nothing
end

function sample_discrete(p)
    n = length(p)
    i = 1
    c = p[1]
    u = rand()
    while c < u && i < n
        c += p[i += 1]
    end

    return i
end

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
    p = similar(θ)
    softmax2!(p, θ)
    # p = softmax(θ)
    return log(p[a])
end

function grad_logpdf!(ψ, π::StatelessSoftmax, θ, a)
    softmax2!(ψ, θ)
    logp = log(ψ[a])
    @. ψ *= -1.0
    ψ[a] += 1.0
    return logp
end


function sample_with_trace!(ψ, action, π::StatelessSoftmax, θ)
    softmax2!(ψ, θ)
    a = sample_discrete(ψ)
    action[1] = a

    logp = log(ψ[a])
    @. ψ *= -1.0
    ψ[a] += 1.0
    return logp
end

function sample_with_trace!(ψ, action, π::StatelessSoftmax, θ, s)
    logp = sample_with_trace!(ψ, action, π, θ)
    return logp
end


function (π::LinearSoftmax)(θ, s)
    p = zeros(size(θ,2))
    # p = softmax(θ' * s)
    mul!(p, θ', s)
    softmax2!(p)
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
    # p = softmax(θ' * s)
    p = zeros(size(θ,2))
    mul!(p, θ', s)
    softmax2!(p)
    return log(p[a])
end

function grad_logpdf!(ψ, π::LinearSoftmax, θ, s, a)
    p = zeros(size(θ,2))
    mul!(p, θ', s)
    softmax2!(p)
    logpa = log(p[a])
    @. p *= -1.0
    # p = -softmax(θ' * s)
    # logpa = log(-p[a])
    p[a] += 1.0
    G = ψ'
    for i in 1:length(p)
        @. G[i, :] = s * p[i]
    end
    return logpa
end


function sample_with_trace!(ψ, action, π::LinearSoftmax, θ, s)
    p = zeros(size(θ,2))
    mul!(p, θ', s)
    softmax2!(p)
    action[1] = sample_discrete(p)
    logp = log(p[action[1]])
    @. p *= -1.0

    # p = softmax(θ' * s)
    # action[1] = rand(Categorical(p))
    # logp = log(p[action[1]])
    # @. p *= -1
    p[action[1]] += 1.0
    G = ψ'
    for i in 1:length(p)
        @. G[i, :] = s * p[i]
    end
    return logp
end
