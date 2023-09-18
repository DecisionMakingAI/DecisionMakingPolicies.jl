
struct LinearPolicyWithBasis{TP, TB} <: LuxCore.AbstractExplicitContainerLayer{(:policy,)}
    ϕ::TB
    policy::TP
end

function (policy::LinearPolicyWithBasis)(s, params, states)
    feats = policy.ϕ(s)
    return policy.policy(feats, params, states)    
end

function logpdf(policy::LinearPolicyWithBasis, s, a, params, states)
    feats = policy.ϕ(s)
    return logpdf(policy.policy, feats, a, params, states)
end

function grad_logpdf(policy::LinearPolicyWithBasis, s, a, params, states)
    feats = policy.ϕ(s)
    return grad_logpdf(policy.policy, feats, a, params, states)
end

function sample_with_trace(policy::LinearPolicyWithBasis, s, params, states)
    feats = policy.ϕ(s)
    return sample_with_trace(policy.policy, feats, params, states)
end