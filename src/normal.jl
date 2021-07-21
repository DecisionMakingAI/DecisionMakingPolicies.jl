
struct StatelessNormal{T} <: AbstractStatelessPolicy where {T}
    μ::T
    σ::T

    function StatelessNormal(::Type{T}, num_actions::Int, std=1.0) where {T}
        μ = zeros(T, num_actions)
        σ = ones(T, num_actions) .* convert.(T, std)
        return new{typeof(μ)}(μ, σ)
    end
    
    function StatelessNormal(num_actions::Int, std=1.0)
        return StatelessNormal(Float64, num_actions, std)
    end

end

function params(π::StatelessNormal)
    return (π.μ, π.σ)
end

struct LinearNormal{T,TS} <: AbstractPolicy  where {T,TS}
    W::T
    σ::TS

    function LinearNormal(::Type{T}, num_features::Int, num_actions::Int, std=1.0) where {T}
        W = zeros(T, (num_features, num_actions))
        σ = ones(T, num_actions) .* convert.(T, std)
        return new{typeof(W),typeof(σ)}(W, σ)
    end

    function LinearNormal(num_features::Int, num_actions::Int, std=1.0)
        return LinearNormal(Float64, num_features, num_actions, std)
    end
end

function params(π::LinearNormal)
    return (π.W, π.σ)
end

struct NormalBuffer{TA, T, TS} <: Any
    action::TA
    μ::T
    ψ::TS

    function NormalBuffer(π::StatelessNormal)
        T = eltype(π.μ)
        n = length(π.μ)
        μ = zeros(T, n)
        ψ = (zero(π.μ), zero(π.σ))
        action = [zeros(T, n)]
        return new{typeof(action),typeof(μ),typeof(ψ)}(action, μ, ψ)
    end

    function NormalBuffer(π::LinearNormal)
        T = eltype(π.W)
        n = size(π.W,2)
        μ = zeros(T, n)
        ψ = (zero(π.W), zero(π.σ))
        action = [zeros(T, n)]
        return new{typeof(action),typeof(μ), typeof(ψ)}(action, μ, ψ)
    end
end

function logpdf_normal(z::T, μ::T, σ::T) where {T}
    tmp = -log(sqrt(T(2.0) * T(π)))
    logp = tmp - (z - μ)^2 / (T(2.0) * σ^2) - log(σ)
    return logp
end

function normal_grad_mu!(G, z, μ, σ)
    @. G = (z - μ) / σ^2
end

function normal_grad_sigma!(G, z, μ, σ)
    T = eltype(z)
    @. G = (T(-1) + ((z - μ) / σ)^2) / σ
end

function linear_normal_gradmu!(G,z,μ,σ,x)
    N = length(z)

    for i in 1:N
        @. G[i, :] = x * (z[i] - μ[i]) / σ[i]^2
    end
end


function (π::StatelessNormal)()
    μ, σ = π.μ, π.σ
    N = length(μ)
    if N > 1
        return MvNormal(μ, σ)
    else
        return Normal(μ[1], σ[1])
    end
end

function (π::StatelessNormal)(buff::NormalBuffer)
    μ, σ = π.μ, π.σ
    N = length(μ)
    if N > 1
        return MvNormal(μ, σ)
    else
        return Normal(μ[1], σ[1])
    end
end

function logpdf(π::StatelessNormal, a)
    μ, σ = π.μ, π.σ
    logp = sum(logpdf_normal.(a, μ, σ))
    return logp
end

function logpdf!(buff::NormalBuffer, π::StatelessNormal, a)
    μ, σ = π.μ, π.σ
    @. buff.μ = logpdf_normal(a, μ, σ)
    logp = sum(buff.μ)
    return logp
end


function grad_logpdf!(ψ, π::StatelessNormal, a)
    μ, σ = π.μ, π.σ
    ψμ, ψσ = ψ[1], ψ[2]
    normal_grad_mu!(ψμ, a, μ, σ)
    normal_grad_sigma!(ψσ, a, μ, σ)

    logp = sum(logpdf_normal.(a, μ, σ))

    return logp, buff.ψ
end

function grad_logpdf!(buff::NormalBuffer, π::StatelessNormal, a)
    μ, σ = π.μ, π.σ
    ψμ, ψσ = buff.ψ[1], buff.ψ[2]
    normal_grad_mu!(ψμ, a, μ, σ)
    normal_grad_sigma!(ψσ, a, μ, σ)
    @. buff.μ = logpdf_normal(a, μ, σ)
    logp = sum(buff.μ)
    return logp, buff.buff.ψ
end

function sample_with_trace!(ψ, action, π::StatelessNormal)
    μ, σ = π.μ, π.σ

    T = eltype(μ)
    @. action[1] = μ + σ * randn((T,))
    logp = grad_logpdf!(ψ, π, action)

    return action[1], logp, buff.ψ
end

function sample_with_trace!(buff::NormalBuffer, π::StatelessNormal)
    μ, σ = π.μ, π.σ
    
    T = eltype(μ)
    @. buff.action[1] = μ + σ * randn((T,))
    ψ = buff.ψ
    logp = grad_logpdf!(ψ, π, buff.action[1])

    return buff.action[1], logp, buff.ψ
end

function (π::LinearNormal)(s)
    W, σ = π.W, π.σ
    μ = W's
    N = length(σ)
    if N > 1
        return MvNormal(μ, σ)
    else
        return Normal(μ[1], σ[1])
    end
end

function (π::LinearNormal)(buff::NormalBuffer, s)
    W, σ = π.W, π.σ
    mul!(buff.μ, W', s)
    N = length(σ)
    if N > 1
        return MvNormal(buff.μ, σ)
    else
        return Normal(buff.μ[1], σ[1])
    end
end

function logpdf(π::LinearNormal, s, a)
    W, σ = π.W, π.σ
    μ = W's
    logp = sum(logpdf_normal.(a, μ, σ))
    return logp
end

function logpdf(π::LinearNormal, s::AbstractMatrix, a)
    W, σ = π.W, π.σ
    μ = W's
    logp = sum(logpdf_normal.(a, μ, σ), dims=1)
    return logp
end

function logpdf!(buff::NormalBuffer, π::LinearNormal, s, a)
    W, σ = π.W, π.σ
    mul!(buff.μ, W', s)
    T = eltype(σ)
    logp = T(0.0)
    for i in 1:length(buff.μ)
        logp += logpdf_normal(a[i], buff.μ[i], σ[i])
    end
    return logp
end


function grad_logpdf!(ψ, π::LinearNormal, s, a)
    W, σ = π.W, π.σ
    μ = W's

    ψw, ψσ = ψ[1]', ψ[2]

    linear_normal_gradmu!(ψw, a, μ, σ, s)
    normal_grad_sigma!(ψσ, a, μ, σ)

    logp = sum(logpdf_normal.(a, μ, σ))

    return logp, ψ
end

function grad_logpdf!(buff::NormalBuffer, π::LinearNormal, s, a)
    W, σ = π.W, π.σ
    mul!(buff.μ,W', s)
    ψ = buff.ψ
    ψw, ψσ = ψ[1]', ψ[2]

    linear_normal_gradmu!(ψw, a, buff.μ, σ, s)
    normal_grad_sigma!(ψσ, a, buff.μ, σ)
    
    T = eltype(σ)
    logp = T(0.0)
    for i in 1:length(buff.μ)
        logp += logpdf_normal(a[i], buff.μ[i], σ[i])
    end

    return logp, ψ
end


function sample_with_trace!(ψ, action, π::LinearNormal, s)
    W, σ = π.W, π.σ
    μ = W's

    T = eltype(σ)
    @. action[1] = μ + σ * randn((T,))

    ψw, ψσ = ψ[1]', ψ[2]

    linear_normal_gradmu!(ψw, action, μ, σ, s)
    normal_grad_sigma!(ψσ, action, μ, σ)

    logp = sum(logpdf_normal.(action, μ, σ))

    return action[1], logp, buff.ψ
end

function sample_with_trace!(buff::NormalBuffer, π::LinearNormal, s)
    μ, action, ψ = buff.μ, buff.action[1], buff.ψ
    W, σ = π.W, π.σ
    mul!(μ, W', s)

    T = eltype(σ)
    @. action = μ + σ * randn((T,))

    ψw, ψσ = ψ[1]', ψ[2]

    linear_normal_gradmu!(ψw, action, μ, σ, s)
    normal_grad_sigma!(ψσ, action, μ, σ)

    logp = T(0.0)
    for i in 1:length(μ)
        logp += logpdf_normal(action[i], μ[i], σ[i])
    end

    return action, logp, buff.ψ
end


