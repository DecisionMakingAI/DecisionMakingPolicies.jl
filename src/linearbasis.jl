
struct LinearPolicyWithBasis{TP, TB} <: AbstractPolicy where {TP, TB}
    π::TP
    ϕ::TB
end

struct LinearWithBasisBuffer{TP,TB} <: Any where {TP,TB}
    policy_buffer::TP
    basis_buffer::TB
end

function params(π::LinearPolicyWithBasis)
    return params(π.π)
end

function (π::LinearPolicyWithBasis)(s)
    feats = π.ϕ(s)
    return π.π(feats)    
end

function (π::LinearPolicyWithBasis)(buff::LinearWithBasisBuffer, s)
    feats = π.ϕ(buff.basis_buff, s)
    return π.π(buff.policy_buff, feats)    
end

function logpdf(π::LinearPolicyWithBasis, s, a)
    feats = π.ϕ(s)
    return logpdf(π.π, feats, a)
end

function logpdf!(buff::LinearWithBasisBuffer, π::LinearPolicyWithBasis, s, a)
    feats = π.ϕ(buff.basis_buff, s)
    return logpdf!(buff.policy_buff, π.π, feats, a)
end

function grad_logpdf!(ψ, π::LinearPolicyWithBasis, s, a)
    feats = π.ϕ(s)
    return grad_logpdf!(ψ, π.π, feats, a)
end

function grad_logpdf!(buff::LinearWithBasisBuffer, π::LinearPolicyWithBasis, s, a)
    feats = π.ϕ(buff.basis_buffer, s)
    return grad_logpdf!(buff.policy_buffer, π.π, feats, a)
end

function sample_with_trace!(ψ, action, π::LinearPolicyWithBasis, s)
    feats = π.ϕ(s)
    return sample_with_trace!(ψ, action, π.π, feats)
end

function sample_with_trace!(buff::LinearWithBasisBuffer, π::LinearPolicyWithBasis, s)
    feats = π.ϕ(buff.basis_buffer, s)
    return sample_with_trace!(buff.policy_buffer, π.π, feats)
end