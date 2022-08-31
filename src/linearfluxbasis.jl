struct LinearPolicyWithFluxBasis{TP, TB} <: AbstractPolicy where {TP, TB}
    π::TP
    ϕ::TB
end


function params(π::LinearPolicyWithFluxBasis)
    return Flux.params(π.ϕ, params(π.π)...)
end

function paramsvec(π::LinearPolicyWithFluxBasis)
    xs = Zygote.Buffer([])
    m = π.ϕ
    fmap(m) do x
        x isa AbstractArray && push!(xs, x)
        return x
    end
    ps = params(π.π)
    for p in ps
        push!(xs, p)
    end
    return vcat(vec.(copy(xs))...)
end

function paramsfromvec!(π::LinearPolicyWithFluxBasis, θ)
    i = 0
    m = π.ϕ
    fmap(m) do x
        x isa AbstractArray || return x
        θx = @view θ[i.+(1:length(x))]
        x .= reshape(θx, size(x))
        i += length(x)
        return x
    end
    length(xs) == i || @warn "Expected $(i) params, got $(length(xs))"
    paramsfromvec!(π.π, @view θ[i+1:end])
    return π
end

function rrule(::typeof(paramsfromvec!), π::LinearPolicyWithFluxBasis, θ)
    i = 0
    m = π.ϕ
    m̄ = fmap(m) do x
        x isa AbstractArray || return x
        θx = @view θ[i.+(1:length(x))]
        x .= reshape(θx, size(x))
        i += length(x)
        return x
    end
    length(xs) == i || @warn "Expected $(i) params, got $(length(xs))"
    paramsfromvec!(π.π, @view θ[i+1:end])

    function paramsfromvec_linearwithflux!(ȳ)
        xs = Zygote.Buffer([])
        dm = ȳ.ϕ
        fmap(dm) do x
            x isa AbstractArray && push!(xs, x)
            return x
        end
        dpi = ȳ.π
        for d in dpi
            push!(xs, p)
        end
        dθ = vcat(vec.(copy(xs))...)
        return NoTangent(), ȳ, dθ
    end
    return π, paramsfromvec_linearwithflux!
end

# function _restructure(m, xs)
#     i = 0
#     m̄ = fmap(m) do x
#         x isa AbstractArray || return x
#         x = reshape(xs[i.+(1:length(x))], size(x))
#         i += length(x)
#         return x
#     end
#     length(xs) == i || @warn "Expected $(i) params, got $(length(xs))"
#     return m̄
# end

# @adjoint function _restructure(m, xs)
#     m̄, numel = _restructure(m, xs), length(xs)
#     function _restructure_pullback(dm)
#         xs′ = destructure(dm)[1]
#         numel == length(xs′) || @warn "Expected $(numel) params, got $(length(xs′))"
#         return (nothing, xs′)
#     end
#     return m̄, _restructure_pullback
# end

# """
#     destructure(m)
# Flatten a model's parameters into a single weight vector.
#     julia> m = Chain(Dense(10, 5, σ), Dense(5, 2), softmax)
#     Chain(Dense(10, 5, σ), Dense(5, 2), softmax)
#     julia> θ, re = destructure(m);
#     julia> θ
#     67-element Vector{Float32}:
#     -0.1407104
#     ...
# The second return value `re` allows you to reconstruct the original network after making
# modifications to the weight vector (for example, with a hypernetwork).
#     julia> re(θ .* 2)
#     Chain(Dense(10, 5, σ), Dense(5, 2), softmax)
# """
# function destructure(m)
#     xs = Zygote.Buffer([])
#     fmap(m) do x
#         x isa AbstractArray && push!(xs, x)
#         return x
#     end
#     return vcat(vec.(copy(xs))...), p -> _restructure(m, p)
# end

function (π::LinearPolicyWithFluxBasis)(s)
    feats = π.ϕ(s)
    return π.π(feats)    
end

function logpdf(π::LinearPolicyWithFluxBasis, s, a)
    feats = π.ϕ(s)
    return logpdf(π.π, feats, a)
end

function logpdf(π::LinearPolicyWithFluxBasis, s)
    feats = π.ϕ(s)
    return logpdf(π.π, feats)
end

function grad_logpdf(π::LinearPolicyWithFluxBasis, s, a)
    ps = params(π)
    logp, gp = Zygote.pullback(()->logpdf(π, s, a), ps)
    tmp = gp(1)
    ψ = [tmp[p] for p in ps]
    return logp, ψ
end

function sample_with_trace(π::LinearPolicyWithFluxBasis, s)
    ps = params(π)
    action = []
    function f!(res, π)
        d = π(s)

        Zygote.ignore() do
            a = rand(d)
            push!(res, a)
        end
        return logpdf(d, res[1])
    end
    logp, back = Zygote.pullback(()->f!(action, π), ps)
    tmp = back(1)
    ψ = [tmp[p] for p in ps]
    return action[1], logp, ψ
end

