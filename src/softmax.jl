struct StatelessSoftmax <: AbstractStatelessPolicy end
struct LinearSoftmax <: AbstractPolicy end

struct SoftmaxBuffer{TA, TP, TS} <: Any
    action::TA
    p::TP
    ψ::TS

    function SoftmaxBuffer(π::StatelessSoftmax, θ)
        T = eltype(θ[1])
        n = length(θ[1])
        p = zeros(T, n)
        ψ = zero.(θ)
        action = zeros(Int, 1)
        return new{typeof(action),typeof(p), typeof(ψ)}(action, p, ψ)
    end

    function SoftmaxBuffer(π::LinearSoftmax, θ)
        T = eltype(θ[1])
        n = size(θ[1], 2)
        p = zeros(T, n)
        ψ = zero.(θ)
        action = zeros(Int, 1)
        return new{typeof(action),typeof(p),typeof(ψ)}(action, p, ψ)
    end
end

function softmax!(out::Vector{T}, x::Vector{T}) where {T}
    @. out = x
    softmax!(out)
    return nothing
end

function softmax!(x::Vector{T}) where {T}
    maxx = maximum(x)
    x .-= maxx
    @. x = exp(x)
    x ./= sum(x)
    return nothing
end

function rrule(::typeof(softmax!), x::Vector{T}) where {T}
    softmax!(x)
    function softmax_pullback(ȳ)
        b = dot(ȳ, x)
        gx = @. (ȳ - b) * x
        return NO_FIELDS, gx
    end
    return x, softmax_pullback
end

function softmax(x::Vector{T}) where {T}
    maxx = maximum(x)
    y = x .- maxx
    @. y = exp(y)
    tot = sum(y)
    z = @. y / tot
    return z
end

function rrule(::typeof(softmax), x::Vector{T}) where {T}
    y = softmax(x)
    function softmax_pullback(ȳ)
        b = dot(ȳ, y)
        gx = @. (ȳ - b) * y
        return NO_FIELDS, gx
    end
    return y, softmax_pullback
end

function sample_discrete(p)
    n = length(p)
    i = 1
    c = p[1]
    u = rand()
    while c < u && i < n
        c += p[i += 1]
    end

    return i
end

function initparams(π::StatelessSoftmax, ::Type{T}, action_dim::Int) where {T}
    N = action_dim
    θ = zeros(T, N)
    return (θ,)
end

function initparams(π::StatelessSoftmax, action_dim::Int)
    return initparams(π, Float64, action_dim)
end

function (π::StatelessSoftmax)(θ)
    p = softmax(θ[1])
    return Categorical(p)
end

function (π::StatelessSoftmax)(buff::SoftmaxBuffer, θ)
    softmax!(buff.p, θ[1])
    return Categorical(buff.p)
end

function (π::StatelessSoftmax)(θ, s)
    return π(θ)
end

function (π::StatelessSoftmax)(buff::SoftmaxBuffer, θ, s)
    return π(buff, θ)
end

function logpdf(π::StatelessSoftmax, θ, a)
    p = similar(θ[1])
    softmax!(p, θ[1])
    return log(p[a])
end

function rrule(::typeof(logpdf), π::StatelessSoftmax, θ, a)
    p = similar(θ[1])
    softmax!(p, θ[1])
    logp = log(p[a])
    T = typeof(logp)
    function logpdf_stateless_softmax_pullback(Ȳ)
        p *= -T(1.0)
        p[a] += T(1.0)
        p *= Ȳ
        return NO_FIELDS, DoesNotExist, (p,), DoesNotExist
    end
    return logp, logpdf_stateless_softmax_pullback
end

function logpdf!(buff::SoftmaxBuffer, π::StatelessSoftmax, θ, a)
    softmax!(buff.p, θ[1])
    return log(buff.p[a])
end


function grad_logpdf!(ψ, π::StatelessSoftmax, θ, a)
    softmax!(ψ[1], θ[1])
    logp = log(ψ[1][a])
    T = eltype(θ[1])
    @. ψ[1] *= -T(1.0)
    ψ[1][a] += T(1.0)
    return logp
end

function grad_logpdf!(buff::SoftmaxBuffer, π::StatelessSoftmax, θ, a)
    return grad_logpdf!(buff.ψ, π, θ, a)
end


function sample_with_trace!(ψ, action, π::StatelessSoftmax, θ)
    softmax!(ψ[1], θ[1])
    a = sample_discrete(ψ[1])
    action[1] = a

    logp = log(ψ[1][a])
    T = typeof(logp)
    @. ψ[1] *= -T(1.0)
    ψ[1][a] += T(1.0)
    return logp
end

function sample_with_trace!(buff::SoftmaxBuffer, π::StatelessSoftmax, θ)
    logp = sample_with_trace!(buff.ψ, buff.action, π, θ)
    return logp
end

function sample_with_trace!(ψ, action, π::StatelessSoftmax, θ, s)
    logp = sample_with_trace!(ψ, action, π, θ)
    return logp
end

function sample_with_trace!(buff::SoftmaxBuffer, π::StatelessSoftmax, θ, s)
    logp = sample_with_trace!(buff.ψ, buff.action, π, θ)
    return logp
end


function (π::LinearSoftmax)(θ, s)
    T = eltype(θ[1])
    p = zeros(T, size(θ[1],2))
    mul!(p, θ[1]', s)
    softmax!(p)
    return Categorical(p)
end

function rrule(::LinearSoftmax, θ, s)
    T = eltype(θ[1])
    p = zeros(T, size(θ[1],2))
    mul!(p, θ[1]', s)
    softmax!(p)
    d = Categorical(p)
    function linear_softmax_dist_pullback(ȳ)
        b = dot(ȳ.p, p)
        dp = @. (ȳ.p - b) * p
        ψ = zero.(θ)
        G = ψ[1]'
        for i in 1:length(dp)
            @. G[i, :] = s * dp[i]
        end
        ds = zero(s)
        mul!(ds, θ[1], dp)
        return NO_FIELDS, ψ, ds
    end
    return d, linear_softmax_dist_pullback
end


function (π::LinearSoftmax)(buff::SoftmaxBuffer, θ, s)
    mul!(buff.p, θ[1]', s)
    softmax!(buff.p)
    return Categorical(buff.p)
end

function initparams(π::LinearSoftmax, ::Type{T}, feature_dim::Int, action_dim::Int) where {T}
    M = feature_dim
    N = action_dim
    θ = zeros(T, (M,N))
    return (θ,)
end

function initparams(π::LinearSoftmax, feature_dim::Int, action_dim::Int)
    return initparams(π, Float64, feature_dim, action_dim)
end

function logpdf(π::LinearSoftmax, θ, s, a)
    T = eltype(θ[1])
    p = zeros(T, size(θ[1],2))
    mul!(p, θ[1]', s)
    softmax!(p)
    return log(p[a])
end

function rrule(::typeof(logpdf), π::LinearSoftmax, θ, s, a)
    T = eltype(θ[1])
    p = zeros(T, size(θ[1],2))
    mul!(p, θ[1]', s)
    softmax!(p)
    logp = log(p[a])
    function logpdf_linear_softmax_pullback(ȳ)
        p *= -T(1.0)
        p[a] += T(1.0)
        p *= ȳ
        ψ = zero.(θ)
        G = ψ[1]'
        for i in 1:length(p)
            @. G[i, :] = s * p[i]
        end
        ds = zero(s)
        mul!(ds, θ[1], p)
        return NO_FIELDS, DoesNotExist, ψ, ds, DoesNotExist
    end
    return logp, logpdf_linear_softmax_pullback
end

function logpdf!(buff::SoftmaxBuffer, π::LinearSoftmax, θ, s, a)
    mul!(buff.p, θ[1]', s)
    softmax!(buff.p)
    return log(buff.p[a])
end

function grad_logpdf!(ψ, π::LinearSoftmax, θ, s, a)
    T = eltype(θ[1])
    p = zeros(T, size(θ[1],2))
    mul!(p, θ[1]', s)
    softmax!(p)
    logpa = log(p[a])
    T = typeof(logpa)
    @. p *= -T(1.0)
    p[a] += T(1.0)
    G = ψ[1]'
    for i in 1:length(p)
        @. G[i, :] = s * p[i]
    end
    return logpa
end

function grad_logpdf!(buff::SoftmaxBuffer, π::LinearSoftmax, θ, s, a)
    mul!(buff.p, θ[1]', s)
    softmax!(buff.p)
    logpa = log(buff.p[a])
    T = typeof(logpa)
    @. buff.p *= -T(1.0)
    buff.p[a] += T(1.0)
    G = buff.ψ[1]'
    for i in 1:length(buff.p)
        @. G[i, :] = s * buff.p[i]
    end
    return logpa
end


function sample_with_trace!(ψ, action, π::LinearSoftmax, θ, s)
    T = eltype(θ[1])
    p = zeros(T, size(θ[1],2))
    mul!(p, θ[1]', s)
    softmax!(p)
    action[1] = sample_discrete(p)
    logp = log(p[action[1]])
    T = typeof(logp)
    @. p *= -T(1.0)

    p[action[1]] += T(1.0)
    G = ψ[1]'
    for i in 1:length(p)
        @. G[i, :] = s * p[i]
    end
    return logp
end

function sample_with_trace!(buff::SoftmaxBuffer, π::LinearSoftmax, θ, s)
    mul!(buff.p, θ[1]', s)
    softmax!(buff.p)
    a = sample_discrete(buff.p)
    buff.action[1] = a
    logp = log(buff.p[a])
    T = typeof(logp)
    @. buff.p *= -T(1.0)

    buff.p[a] += T(1.0)
    G = buff.ψ[1]'
    for i in 1:length(buff.p)
        @. G[i, :] = s * buff.p[i]
    end
    return logp
end
