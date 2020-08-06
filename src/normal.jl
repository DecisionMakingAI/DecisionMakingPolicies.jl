# import Distributions: Normal, MvNormal

struct StatelessNormal <: AbstractStatelessPolicy end
struct LinearNormal <: AbstractPolicy end

export StatelessNormal, LinearNormal


function logpdf_normal(z::T, μ::T, σ::T) where {T}
    tmp = -log(sqrt(T(2.0) * T(π)))
    logp = tmp - (z - μ)^2 / (T(2.0) * σ^2) - log(σ)
    return logp
end

function normal_grad_mu!(G, z, μ, σ)
    @. G = (z - μ) / σ^2
end

function normal_grad_sigma!(G, z, μ, σ)
    @. G = (-1 + ((z - μ) / σ)^2) / σ
end

function linear_normal_gradmu!(G,z,μ,σ,x)
    N = length(z)

    for i in 1:N
        @. G[i, :] = x * (z[i] - μ[i]) / σ[i]^2
    end
end


function (π::StatelessNormal)(θ)
    μ = θ[1]
    σ = θ[2]
    N = length(μ)
    if N > 1
        return MvNormal(μ, σ)
    else
        return Normal(μ[1], σ[1])
    end
end

function initparams(π::StatelessNormal, ::Type{T}, action_dim::Int, std=1.0) where {T}
    N = action_dim
    μ = zeros(T, N)
    σ = ones(T, N) .* convert.(T, std)
    return (μ, σ)
end

function initparams(π::StatelessNormal, action_dim::Int, std=1.0)
    return initparams(π, Float64, action_dim, std)
end



function logpdf(π::StatelessNormal, θ, a)
    μ = θ[1]
    σ = θ[2]
    logp = sum(logpdf_normal.(a, μ, σ))
    return logp
end


function grad_logpdf!(ψ, π::StatelessNormal, θ, a)
    μ = θ[1]
    σ = θ[2]
    ψμ = ψ[1]
    ψσ = ψ[2]
    normal_grad_mu!(ψμ, a, μ, σ)
    normal_grad_sigma!(ψσ, a, μ, σ)

    logp = sum(logpdf_normal.(a, μ, σ))

    return logp
end

function sample_with_trace!(ψ, action, π::StatelessNormal, θ)
    μ = θ[1]
    σ = θ[2]

    T = eltype(μ)
    @. action = μ + σ * randn((T,))
    logp = grad_logpdf!(ψ, π, θ, action)

    return logp
end

function (π::LinearNormal)(θ, s)
    W = θ[1]
    μ = W's
    σ = θ[2]
    N = length(σ)
    if N > 1
        return MvNormal(μ, σ)
    else
        return Normal(μ[1], σ[1])
    end
end

function initparams(π::LinearNormal, ::Type{T}, feature_dim::Int, action_dim::Int, std=1.0) where {T}
    M = feature_dim
    N = action_dim
    W = zeros(T, (M,N))
    σ = ones(T, N) .* convert.(T, std)
    return (W, σ)
end

function initparams(π::LinearNormal, feature_dim::Int, action_dim::Int, std=1.0)
    return init_params(π, Float64, feature_dim, action_dim, std)
end

function logpdf(π::LinearNormal, θ, s, a)
    W = θ[1]
    μ = W's
    σ = θ[2]
    logp = sum(logpdf_normal.(a, μ, σ))
    return logp
end


function grad_logpdf!(ψ, π::LinearNormal, θ, s, a)
    W = θ[1]
    μ = W's
    σ = θ[2]

    ψw = ψ[1]'
    ψσ = ψ[2]

    linear_normal_gradmu!(ψw, a, μ, σ, s)
    normal_grad_sigma!(ψσ, a, μ, σ)

    logp = sum(logpdf_normal.(a, μ, σ))

    return logp
end


function sample_with_trace!(ψ, action, π::LinearNormal, θ, s)
    W = θ[1]
    μ = W's
    σ = θ[2]

    T = eltype(σ)
    @. action = μ + σ * randn((T,))

    ψw = ψ[1]'
    ψσ = ψ[2]

    linear_normal_gradmu!(ψw, action, μ, σ, s)
    normal_grad_sigma!(ψσ, action, μ, σ)

    logp = sum(logpdf_normal.(action, μ, σ))

    return logp
end
