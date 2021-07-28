struct LinearPolicyWithFluxBasis{TP, TB} <: AbstractPolicy where {TP, TB}
    π::TP
    ϕ::TB
end


function params(π::LinearPolicyWithFluxBasis)
    return Flux.params(π.ϕ, params(π.π)...)
end

function (π::LinearPolicyWithFluxBasis)(s)
    feats = π.ϕ(s)
    return π.π(feats)    
end

function logpdf(π::LinearPolicyWithFluxBasis, s, a)
    feats = π.ϕ(s)
    return logpdf(π.π, feats, a)
end

function grad_logpdf(π::LinearPolicyWithFluxBasis, s, a)
    ps = params(π)
    logp, gp = Zygote.pullback(()->logpdf(π, s, a), ps)
    tmp = gp(1)
    ψ = [tmp[p] for p in ps]
    return logp, ψ
end

