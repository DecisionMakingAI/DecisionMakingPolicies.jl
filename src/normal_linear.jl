

function (π::LinearNormal)(s)
    W, σ = π.W, π.σ
    μ = W's
    N = length(σ)
    if N > 1
        return MvNormal(μ, Diagonal(map(abs2,σ)))
    else
        return Normal(μ[1], σ[1])
    end
end

function (π::LinearNormal)(buff::NormalBuffer, s)
    W, σ = π.W, π.σ
    mul!(buff.μ, W', s)
    N = length(σ)
    if N > 1
        return MvNormal(buff.μ, Diagonal(map(abs2,σ)))
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
    μ = W' * s
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

function grad_logpdf(π::LinearNormal, s, a)
    G = zero(π.W), zero(π.σ)
    logpa, ψ = grad_logpdf!(G, π, s, a)
    return logpa, ψ
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
    @. action = μ + σ * randn((T,))

    ψw, ψσ = ψ[1]', ψ[2]

    linear_normal_gradmu!(ψw, action, μ, σ, s)
    normal_grad_sigma!(ψσ, action, μ, σ)

    logp = sum(logpdf_normal.(action, μ, σ))

    return action, logp, ψ
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


