
struct StatelessSoftmax{T} <: AbstractStatelessPolicy where {T}
    θ::T

    function StatelessSoftmax(::Type{T}, num_actions::Int) where {T}
        θ = zeros(T, num_actions)
        return new{typeof(θ)}(θ)
    end

    function StatelessSoftmax(num_actions::Int) where {T}
        return StatelessSoftmax(Float64, num_actions)
    end
end

function params(π::StatelessSoftmax)
    return (π.θ,)
end

function paramsvec(π::StatelessSoftmax)
    return copy(vec(π.θ))
end

function paramsfromvec!(π::StatelessSoftmax, θ)
    @. π.θ = θ
    return π
end

function rrule(::typeof(paramsfromvec!), π::StatelessSoftmax, θ)
    @. π.θ = θ
    function paramsfromvec_statelesssoftmax!(ȳ)
        dθ = ȳ.θ
        return NoTangent(), ȳ, dθ
    end
    return π, paramsfromvec_statelesssoftmax!
end

struct LinearSoftmax{T} <: AbstractPolicy where {T}
    θ::T

    function LinearSoftmax(::Type{T}, num_features::Int, num_actions::Int) where {T}
        θ = zeros(T, (num_features, num_actions))
        return new{typeof(θ)}(θ)
    end

    function LinearSoftmax(num_features::Int, num_actions::Int)
        return LinearSoftmax(Float64, num_features, num_actions)
    end
end

function params(π::LinearSoftmax)
    return (π.θ,)
end

function paramsvec(π::LinearSoftmax)
    return copy(vec(π.θ))
end

function paramsfromvec!(π::LinearSoftmax, θ)
    @. π.θ = θ
    return π
end

function rrule(::typeof(paramsfromvec!), π::LinearSoftmax, θ)
    @. π.θ = θ
    function paramsfromvec_softmax!(ȳ)
        dθ = ȳ.θ
        return NoTangent(), ȳ, dθ
    end
    return π, paramsfromvec_softmax!
end

struct SoftmaxBuffer{TA, TP, TS} <: Any
    action::TA
    p::TP
    ψ::TS

    function SoftmaxBuffer(π::StatelessSoftmax)
        T = eltype(π.θ)
        n = length(π.θ)
        p = zeros(T, n)
        ψ = (zero(π.θ),)
        action = zeros(Int, 1)
        return new{typeof(action),typeof(p), typeof(ψ)}(action, p, ψ)
    end

    function SoftmaxBuffer(π::LinearSoftmax)
        T = eltype(π.θ)
        n = size(π.θ, 2)
        p = zeros(T, n)
        ψ = (zero(π.θ),)
        action = zeros(Int, 1)
        return new{typeof(action),typeof(p),typeof(ψ)}(action, p, ψ)
    end
end

function softmax!(out::AbstractVector, x::AbstractVector)
    @. out = x
    softmax!(out)
    return nothing
end

function softmax!(x::AbstractVector) 
    maxx = maximum(x)
    x .-= maxx
    @. x = exp(x)
    x ./= sum(x)
    return nothing
end

function rrule(::typeof(softmax!), x::AbstractVector)
    softmax!(x)
    function softmax_pullback(ȳ)
        b = dot(ȳ, x)
        gx = @. (ȳ - b) * x
        return NoTangent(), gx
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
        return NoTangent(), gx
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

function (π::StatelessSoftmax)()
    p = softmax(π.θ)
    return Categorical(p)
end

function (π::StatelessSoftmax)(buff::SoftmaxBuffer)
    softmax!(buff.p, π.θ)
    return Categorical(buff.p)
end

function (π::StatelessSoftmax)(s)
    return π()
end

function (π::StatelessSoftmax)(buff::SoftmaxBuffer, s)
    return π(buff)
end

function logpdf(π::StatelessSoftmax, a)
    p = similar(π.θ)
    softmax!(p, π.θ)
    return log(p[a])
end

function rrule(::typeof(logpdf), π::StatelessSoftmax, a)
    p = similar(π.θ)
    softmax!(p, π.θ)
    logp = log(p[a])
    T = typeof(logp)
    function logpdf_stateless_softmax_pullback(Ȳ)
        p *= -T(1.0)
        p[a] += T(1.0)
        p *= Ȳ
        dθ = Tangent{typeof(π)}(θ=p)
        return NoTangent(), dθ, NoTangent()
    end
    return logp, logpdf_stateless_softmax_pullback
end

function logpdf!(buff::SoftmaxBuffer, π::StatelessSoftmax, a)
    softmax!(buff.p, π.θ)
    return log(buff.p[a])
end

function logpdf!(buff::SoftmaxBuffer, π::StatelessSoftmax, s, a)
    return logpdf!(buff, π, a)
end


function grad_logpdf!(ψ, π::StatelessSoftmax, a)
    softmax!(ψ[1], π.θ)
    logp = log(ψ[1][a])
    T = eltype(π.θ)
    @. ψ[1] *= -T(1.0)
    ψ[1][a] += T(1.0)
    return logp, ψ
end

function grad_logpdf!(buff::SoftmaxBuffer, π::StatelessSoftmax, a)
    return grad_logpdf!(buff.ψ, π, a)
end

function grad_logpdf!(buff::SoftmaxBuffer, π::StatelessSoftmax, s, a)
    return grad_logpdf!(buff.ψ, π, a)
end


function sample_with_trace!(ψ, action, π::StatelessSoftmax)
    softmax!(ψ[1], π.θ)
    a = sample_discrete(ψ[1])
    action[1] = a

    logp = log(ψ[1][a])
    T = typeof(logp)
    @. ψ[1] *= -T(1.0)
    ψ[1][a] += T(1.0)
    return a, logp, ψ
end

function sample_with_trace!(buff::SoftmaxBuffer, π::StatelessSoftmax)
    return sample_with_trace!(buff.ψ, buff.action, π)
end

function sample_with_trace!(ψ, action, π::StatelessSoftmax, s)
    return sample_with_trace!(ψ, action, π)
end

function sample_with_trace!(buff::SoftmaxBuffer, π::StatelessSoftmax, s)
    return sample_with_trace!(buff.ψ, buff.action, π)
end


function (π::LinearSoftmax)(s)
    T = eltype(π.θ)
    p = zeros(T, size(π.θ,2))
    mul!(p, π.θ', s)
    softmax!(p)
    return Categorical(p)
end

function rrule(π::LinearSoftmax, s)
    T = eltype(π.θ)
    p = zeros(T, size(π.θ,2))
    mul!(p, π.θ', s)
    softmax!(p)
    d = Categorical(p)
    function linear_softmax_dist_pullback(ȳ)
        b = dot(ȳ.p, p)
        dp = @. (ȳ.p - b) * p
        ψ = zero(π.θ)
        G = ψ'
        for i in 1:length(dp)
            @. G[i, :] = s * dp[i]
        end
        ds = zero(s)
        mul!(ds, π.θ, dp)
        dθ = Tangent{typeof(π)}(θ=ψ)
        return dθ, ds
    end
    return d, linear_softmax_dist_pullback
end


function (π::LinearSoftmax)(buff::SoftmaxBuffer, s)
    mul!(buff.p, π.θ', s)
    softmax!(buff.p)
    return Categorical(buff.p)
end

function logpdf(π::LinearSoftmax, s, a)
    return logpdf_softmax(π.θ, s, a)
end

function logpdf_softmax(θ, s, a)
    T = eltype(θ)
    p = zeros(T, size(θ,2))
    mul!(p, θ', s)
    softmax!(p)
    return log(p[a])
end

function rrule(::typeof(logpdf_softmax), θ, s, a)
    T = eltype(θ)
    p = zeros(T, size(θ,2))
    mul!(p, θ', s)
    softmax!(p)
    logp = log(p[a])

    function logpdf_linear_softmax_pullback(ȳ)
        p *= -T(1.0)
        p[a] += T(1.0)
        p *= ȳ
        ψ = zero(θ)
        G = ψ'
        for i in 1:length(p)
            @. G[i, :] = s * p[i]
        end
        ds = zero(s)
        mul!(ds, θ, p)
        return NoTangent(), ψ, ds, NoTangent()
    end
    return logp, logpdf_linear_softmax_pullback
end

function logpdf_softmax(θ, s::AbstractMatrix, a)
    T = eltype(θ)
    num_a = size(θ,2)
    n = size(s,2)
    p = zeros(T, (num_a, n))
    mul!(p, θ', s)
    logps = zeros(T, (1,n))
    for i in 1:n
        p_i = @view p[:, i]
        softmax!(p_i)
        logps[1,i] = log(p_i[a[i]])
    end
    return logps
end

function rrule(::typeof(logpdf_softmax), θ, s::AbstractMatrix, a)
    T = eltype(θ)
    num_a = size(θ,2)
    n = size(s,2)
    p = zeros(T, (num_a, n))
    mul!(p, θ', s)
    logps = zeros(T, (1,n))
    for i in 1:n
        p_i = @view p[:, i]
        softmax!(p_i)
        logps[1,i] = log(p_i[a[1,i]])
    end

    function logpdf_linear_softmax_batch_pullback(ȳ)
        ψ = zero(θ)
        G = ψ'
        for i in 1:n
            p[:, i] *= -one(T)
            p[a[1, i], i] += one(T)
            p[:, i] *= ȳ[1, i]
            for j in 1:num_a
                x = @view s[:, i]
                @. G[j, :] += x * p[j,i]
            end
        end
        ds = zero(s)
        mul!(ds, θ, p)
        return NoTangent(), ψ, ds, ZeroTangent()
    end
    return logps, logpdf_linear_softmax_batch_pullback
end

function logpdf!(buff::SoftmaxBuffer, π::LinearSoftmax, s, a)
    mul!(buff.p, π.θ', s)
    softmax!(buff.p)
    return log(buff.p[a])
end

function grad_logpdf!(ψ, π::LinearSoftmax, s, a)
    T = eltype(π.θ)
    p = zeros(T, size(π.θ,2))
    mul!(p, π.θ', s)
    softmax!(p)
    logpa = log(p[a])
    T = typeof(logpa)
    @. p *= -T(1.0)
    p[a] += T(1.0)
    G = ψ[1]'
    for i in 1:length(p)
        @. G[i, :] = s * p[i]
    end
    return logpa, ψ
end

function grad_logpdf!(buff::SoftmaxBuffer, π::LinearSoftmax, s, a)
    mul!(buff.p, π.θ', s)
    softmax!(buff.p)
    logpa = log(buff.p[a])
    T = typeof(logpa)
    @. buff.p *= -T(1.0)
    buff.p[a] += T(1.0)
    G = buff.ψ[1]'
    for i in 1:length(buff.p)
        @. G[i, :] = s * buff.p[i]
    end
    return logpa, buff.ψ
end


function sample_with_trace!(ψ, action, π::LinearSoftmax, s)
    T = eltype(π.θ)
    p = zeros(T, size(π.θ,2))
    mul!(p, π.θ', s)
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
    return action[1], logp, ψ
end

function sample_with_trace!(buff::SoftmaxBuffer, π::LinearSoftmax, s)
    mul!(buff.p, π.θ', s)
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
    return a, logp, buff.ψ
end
