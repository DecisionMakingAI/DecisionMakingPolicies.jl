struct FluxPolicy{TM,TD} <: AbstractPolicy
    model::TM
    distribution::TD
end

struct FluxBuffer{TA,TP} <:Any
    action::TA
    ψ::TP

    function FluxBuffer(π, s)
        ψ = zero.(params(π))
        d = π(s)
        a = rand(d)

        action = [zero(a)]
        return new{typeof(action), typeof(ψ)}(action, ψ)
    end 
end

function (π::FluxPolicy)(s)
    out = π.model(s)
    d = π.distribution(out)
    return d
end

function params(π::FluxPolicy)
    return Flux.params(π.model)
end

function logpdf(π::FluxPolicy, s, a)
    d = π(s)
    logp = logpdf(d, a)
    return logp
end

# function grad_logpdf(π::FluxPolicy, s, a)
#     ps = params(π)
#     logp, g = Zygote.pullback(ps->logpdf(π(s), a), ps)
#     return logp, g(1)[1].grads
# end

function grad_logpdf!(buff::FluxBuffer, π::FluxPolicy, s, a)
    ps = Flux.params(π.model)
    logp = logpdf(π, s, a)
    logp, back = Zygote.pullback(()->logpdf(π,s,a), ps)
    g = back(1).grads
    for (p, ψ) in zip(ps, buff.ψ)
        @. ψ = g[p]
    end
    return logp, buff.ψ
end

function sample_with_trace!(buff::FluxBuffer, π::FluxPolicy, s)
    function f!(res, π, s)
        d = π(s)

        Zygote.ignore() do
            a = rand(d)
            if length(a) == 1
                res[1] = a
            else
                res .= a
            end
        end
        if length(d) > 2
            return logpdf(d, res)
        else
            return logpdf(d, res[1])
        end
    end
    ps = Flux.params(π.model)
    action = buff.action
    logp, back = Zygote.pullback(()->f!(action, π, s), ps)
    g = back(1).grads
    for (p, ψ) in zip(ps, buff.ψ)
        @. ψ = g[p]
    end
    return action[1], logp, buff.ψ
end
