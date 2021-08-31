
struct LinearPolicyWithBasis{TP, TB} <: AbstractPolicy where {TP, TB}
    π::TP
    ϕ::TB
end

function params(π::LinearPolicyWithBasis)
    return params(π.π)
end

function (π::LinearPolicyWithBasis)(s)
    feats = π.ϕ(s)
    return π.π(feats)    
end

function logpdf(π::LinearPolicyWithBasis, s, a)
    feats = π.ϕ(s)
    return logpdf(π.π, feats, a)
end


function grad_logpdf!(ψ, π::LinearPolicyWithBasis, s, a)
    feats = π.ϕ(s)
    return grad_logpdf!(ψ, π.π, feats, a)
end

function grad_logpdf(π::LinearPolicyWithBasis, s, a)
    feats = π.ϕ(s)
    logp, ψ = grad_logpdf(π.π, feats, a)
    return logp, ψ
end

function sample_with_trace!(ψ, action, π::LinearPolicyWithBasis, s)
    feats = π.ϕ(s)
    return sample_with_trace!(ψ, action, π.π, feats)
end

function sample_with_trace(π::LinearPolicyWithBasis, s)
    feats = π.ϕ(s)
    return sample_with_trace(π.π, feats)
end