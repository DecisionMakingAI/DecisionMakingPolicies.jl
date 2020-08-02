import Distributions: Normal, MvNormal


struct LinearNormal <: AbstractPolicy end

export LinearNormal

function (π::LinearNormal)(θ, s)
    W = θ[1]
    μ = W's
    σ = θ[2]
    N = length(σ)
    if N > 1
        return MvNormal(μ, σ)
    else
        return Normal(μ, σ)
    end
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

    Gw = ψ[1]'
    Gσ = ψ[2]

    N = length(a)

    for i in 1:N
        @. Gw[i, :] = s * (a[i] - μ[i]) / σ[i]^2
    end

    @. Gσ = (-1 + ((a - μ) / σ)^2) / σ

    logp = sum(logpdf_normal.(a, μ, σ))

    return logp
end


function sample_with_trace!(ψ, action, π::LinearNormal, θ, s)
    W = θ[1]
    μ = W's
    σ = θ[2]

    T = eltype(σ)
    @. action = μ + σ * randn((T,))

    Gw = ψ[1]'
    Gσ = ψ[2]

    N = length(action)

    for i in 1:N
        @. Gw[i, :] = s * (action[i] - μ[i]) / σ[i]^2
    end

    @. Gσ = (-1 + ((action - μ) / σ)^2) / σ

    logp = sum(logpdf_normal.(action, μ, σ))
    return logp
end
