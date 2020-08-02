
struct StatelessSoftmax <: AbstractStatelessPolicy end
struct StatelessNormal <: AbstractStatelessPolicy end

export StatelessSoftmax, StatelessNormal

function (π::StatelessSoftmax)(θ)
    p = softmax(θ)
    return Categorical(p)
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
    action[1] = rand(Categorical(p))

    logpa = log(ψ[a])
    @. ψ *= -1.0
    ψ[a] += 1.0
    return logp
end


function (π::StatelessNormal)(θ)
    N = round(Int,length(θ)/2)
    if N > 1
        return MvNormal(θ[1:N], θ[N+1:end])
    else
        return Normal(θ[1], θ[2])
    end
end

function logpdf_normal(z, μ, σ)
    logp = -log(sqrt(2.0 * π)) - (z - μ)^2 / (2.0 * σ^2) - log(σ)
    return logp
end


function logpdf(π::StatelessNormal, θ, a)
    N = length(a)
    μ = view(θ, 1:N)
    σ = view(θ, N+1:length(θ))
    logp = sum(logpdf_normal.(a, μ, σ))
    return logp
end

function grad_logpdf!(ψ, π::StatelessNormal, θ, a)
    N = length(a)
    μ = view(θ, 1:N)
    σ = view(θ, N+1:length(θ))
    @. ψ[1:N] = (a - μ) / σ^2
    @. ψ[N+1:end] = (-1 + ((a - μ) / σ)^2) / σ

    logp = sum(logpdf_normal.(a, μ, σ))

    return logp
end

function sample_with_trace!(ψ, action, π::StatelessNormal, θ)
    N = length(action)
    μ = view(θ, 1:N)
    σ = view(θ, N+1:length(θ))
    T = eltype(θ)
    @. action = μ + σ * randn((T,))

    @. ψ[1:N] = (action - μ) / σ^2
    @. ψ[N+1:end] = (-1 + ((action - μ) / σ)^2) / σ

    logp = sum(logpdf_normal.(action, μ, σ))

    return logp
end
