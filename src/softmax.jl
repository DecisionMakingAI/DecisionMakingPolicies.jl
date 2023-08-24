
function softmax!(out::AbstractVector, x::AbstractVector)
    @. out = x
    softmax!(out)
    return nothing
end

function softmax!(x::AbstractVector) 
    maxx = maximum(x)
    x .-= maxx
    @. x = exp(x)
    tot = sum(x)
    @. x /= tot
    return x
end

function rrule(::typeof(softmax!), x::AbstractVector)
    y = softmax!(x)
    function softmax_pullback(ȳ)
        dx = @thunk begin
            gx = zero(x)
            b = dot(ȳ, y)
            @. gx = (ȳ - b) * y
            return gx
        end
        return NoTangent(), gx
    end
    return y, softmax_pullback
end

function softmax(x::AbstractVector)
    maxx = maximum(x)
    y = x .- maxx
    @. y = exp(y)
    tot = sum(y)
    z = @. y / tot
    return z
end

function rrule(::typeof(softmax), x::AbstractVector)
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

@non_differentiable sample_discrete(Any)

struct LinearSoftmax{B,F1} <: LuxCore.AbstractExplicitLayer
    in_dims::Int
    num_actions::Int
    init_weights::F1
end

function LinearSoftmax(in_dims::Int, num_actions::Int; init_weights=LuxCore.zeros, buffered::Bool=false)
    return LinearSoftmax{typeof(Val(buffered)), typeof(init_weights)}(in_dims, num_actions, init_weights)
end

LuxCore.zeros(rng::Random.AbstractRNG, size...) = LuxCore.zeros(size...)

function LuxCore.initialparameters(rng::Random.AbstractRNG, layer::LinearSoftmax)
    return (; weight=layer.init_weights(rng, layer.in_dims, layer.num_actions))
end

LuxCore.initialstates(rng::Random.AbstractRNG, layer::LinearSoftmax) = NamedTuple()

function LuxCore.initialstates(rng::Random.AbstractRNG, layer::LinearSoftmax{Val{true}})
    T = eltype(layer.init_weights(rng, 1,1))
    action = zeros(Int, 1)
    p = zeros(T, layer.num_actions)
    ψ = (;weight = zeros(T, layer.in_dims, layer.num_actions)) |> ComponentArray
    
    return (;action=action, p=p, ψ=ψ)
end

LuxCore.parameterlength(layer::LinearSoftmax) = layer.in_dims * layer.num_actions

LuxCore.statelength(layer::LinearSoftmax) = 0
LuxCore.statelength(layer::LinearSoftmax{Val{true}}) = layer.in_dims * layer.num_actions + layer.num_actions + 1

function pdf_softmax(θ, s::AbstractMatrix)
    T = eltype(θ)
    num_a = size(θ,2)
    n = size(s,2)
    p = zeros(T, (num_a, n))
    mul!(p, θ', s)
    softmax_batch!(p)
    return p
end

function rrule(::typeof(pdf_softmax), θ, s::AbstractMatrix)
    T = eltype(θ)
    num_a = size(θ,2)
    n = size(s,2)
    p = zeros(T, (num_a, n))
    mul!(p, θ', s)
    softmax_batch!(p)

    function pdf_linear_softmax_batch_pullback(ȳ)
        dp = copy(p)
        y = unthunk(ȳ)
        for i in 1:n
            ȳi = @view y[:, i]
            dpi = @view dp[:, i]
            b = dot(ȳi, dpi)
            @. dpi *= (ȳi - b)
        end
        dθ = @thunk begin
            ψ = zero(θ)
            mul!(ψ, s, dp')
            # G = ψ'
            # for j in 1:num_a
            #     x = @view s[:, i]
            #     @. G[j, :] += x * dp[j,i]
            # end
        end
        ds = @thunk begin 
            dx = zero(s)
            mul!(dx, θ, dp)
            return dx
        end
        return NoTangent(), dθ, ds
    end
    return p, pdf_linear_softmax_batch_pullback
end

function pdf_softmax(θ, s::AbstractVector)
    T = eltype(θ)
    num_a = size(θ,2)
    p = zeros(T, num_a)
    mul!(p, θ', s)
    softmax!(p)
    return p
end

function pdf_softmax!(p, θ, s::AbstractVector)
    mul!(p, θ', s)
    p = softmax!(p)
    return p
end

function rrule(::typeof(pdf_softmax), θ, s::AbstractVector)
    T = eltype(θ)
    p = zeros(T, size(θ,2))
    mul!(p, θ', s)
    softmax!(p)
    function linear_softmax_pullback(ȳ)
        b = dot(ȳ, p)
        dp = @. (ȳ - b) * p
        dθ = @thunk begin
            ψ = zero(θ)
            # G = ψ'
            # for i in eachindex(dp)
            #     @. G[i, :] = s * dp[i]
            # end
            mul!(ψ, s, dp')
            return ψ
        end
        ds = @thunk begin 
            dx = zero(s)
            mul!(dx, θ, dp)
            return dx
        end
        return NoTangent(), dθ, ds
    end
    return p, linear_softmax_pullback
end

function rrule(::typeof(pdf_softmax!), p, θ, s::AbstractVector)
    T = eltype(θ)
    mul!(p, θ', s)
    p = softmax!(p)
    function linear_softmax_pullback(ȳ)
        b = dot(ȳ, p)
        dp = @. (ȳ - b) * p
        dθ = @thunk begin
            ψ = zero(θ)
            # G = ψ'
            # for i in eachindex(dp)
            #     @. G[i, :] = s * dp[i]
            # end
            mul!(ψ, s, dp')
            return ψ
        end
        ds = @thunk begin 
            dx = zero(s)
            mul!(dx, θ, dp)
            return dx
        end
        return NoTangent(), dθ, ds
    end
    return p, linear_softmax_pullback
end

function softmax_batch!(p::AbstractMatrix{T}) where {T}
    for i in axes(p, 2)
        p_i = @view p[:, i]
        softmax!(p_i)
    end
    return nothing
end


function make_buff(layer::LinearSoftmax, params)
    T = eltype(params.weight)
    action = zeros(Int, 1)
    p = zeros(T, size(params.weight, 2))
    dW = zero(params.weight)
    ψ = (;weight=dW) |> ComponentArray
    buff = (action=action, p=p, ψ=ψ)
    return buff
end

@non_differentiable make_buff(LinearSoftmax, Any)

function (layer::LinearSoftmax)(x::AbstractMatrix, params, states)
    return pdf_softmax(params.weight, x), states
end

function (layer::LinearSoftmax)(x::AbstractVector, params, states)
    p = pdf_softmax(params.weight, x)
    return Categorical(p), states
end

function (layer::LinearSoftmax{Val{true}})(x::AbstractVector, params, states)
    p = pdf_softmax!(states.p, params.weight, x)
    return Categorical(p), states
end

function logpdf_softmax(θ, s::AbstractMatrix, a)
    T = eltype(θ)
    num_a = size(θ,2)
    n = size(s,2)
    p = zeros(T, (num_a, n))
    mul!(p, θ', s)
    logps = softmax_batch_out!(p, a)
    return logps
end

function logpdf_softmax(buff, θ, s::AbstractVector, a)
    mul!(buff.p, θ', s)
    softmax!(buff.p)
    logp = log(buff.p[a])
    return logp
end

function logpdf_softmax(θ, s::AbstractVector, a)
    p = zeros(eltype(θ), size(θ,2))
    mul!(p, θ', s)
    softmax!(p)
    logp = log(p[a])
    return logp
end

function softmax_batch_out!(p::AbstractMatrix{T}, a::AbstractVector{Int}) where {T}
    n = size(p, 2)
    logps = zeros(T, n)
    for i in 1:n
        p_i = @view p[:, i]
        softmax!(p_i)
        logps[i] = log(p_i[a[i]])
    end
    return logps
end

function rrule(::typeof(logpdf_softmax), buff, θ, s::AbstractVector, a::Int)
    T = eltype(θ)
    # num_a = size(θ,2)
    # p = zeros(T, num_a)
    # mul!(p, θ', s)
    # softmax!(p)
    # logp = log(p[a])
    mul!(buff.p, θ', s)
    softmax!(buff.p)
    logp = log(buff.p[a])

    function logpdf_linear_softmax_pullback(ȳ)
        dp = copy(buff.p)
        @. dp *= -one(T)
        dp[a] += one(T)
        @. dp *= ȳ
        dθ = @thunk begin
            ψ = buff.ψ.weight
            G = ψ'
            for i in eachindex(dp)
                @. G[i, :] = s * dp[i]
            end
            return ψ
        end
        ds = @thunk begin 
            dx = zero(s)
            mul!(dx, θ, dp)
            return dx
        end
        return NoTangent(), NoTangent(), dθ, ds, NoTangent()
    end
    return logp, logpdf_linear_softmax_pullback
end

function rrule(::typeof(logpdf_softmax), θ, s::AbstractVector, a::Int)
    T = eltype(θ)
    num_a = size(θ,2)
    p = zeros(T, num_a)
    mul!(p, θ', s)
    softmax!(p)
    logp = log(p[a])

    function logpdf_linear_softmax_pullback(ȳ)
        dp = copy(p)
        @. dp *= -one(T)
        dp[a] += one(T)
        @. dp *= ȳ
        dθ = @thunk begin
            ψ = zero(θ)
            G = ψ'
            for i in eachindex(dp)
                @. G[i, :] = s * dp[i]
            end
            return ψ
        end
        ds = @thunk begin 
            dx = zero(s)
            mul!(dx, θ, dp)
            return dx
        end
        return NoTangent(), dθ, ds, NoTangent()
    end
    return logp, logpdf_linear_softmax_pullback
end

function rrule(::typeof(logpdf_softmax), θ, s::AbstractMatrix, a)
    T = eltype(θ)
    num_a = size(θ,2)
    n = size(s,2)
    p = zeros(T, (num_a, n))
    mul!(p, θ', s)
    logps = softmax_batch_out!(p, a)

    function logpdf_linear_softmax_batch_pullback(ȳ)
        dp = copy(p)
        @. dp *= -one(T)
        for i in 1:n
            dp[a[i], i] += one(T)
            @. dp[:, i] *= ȳ[i]
        end
        dθ = @thunk begin
            ψ = zero(θ)
            # G = ψ'
            # for i in axes(dp, 2)
            #     for j in 1:num_a
            #         x = @view s[:, i]
            #         @. G[j, :] += x * dp[j,i]
            #     end
            # end
            mul!(ψ, s, dp')
        end
        ds = @thunk begin 
            dx = zero(s)
            mul!(dx, θ, dp)
            return dx
        end
        return NoTangent(), dθ, ds, NoTangent()
    end
    return logps, logpdf_linear_softmax_batch_pullback
end

function logpdf(layer::LinearSoftmax, x, a, params, states)
    return logpdf_softmax(params.weight, x, a), states
end

function logpdf(layer::LinearSoftmax, x::AbstractVector, a, params, states)
    buff = make_buff(layer, params)
    return logpdf_softmax(buff, params.weight, x, a), states
end

function logpdf(layer::LinearSoftmax{Val{true}}, s::AbstractVector, a, params, states)
    return logpdf_softmax(states, params.weight, s, a), states
end

function grad_logpdf_linear_softmax!(buff, θ, s::AbstractVector, a::Int)
    mul!(buff.p, θ', s)
    softmax!(buff.p)
    logpa = log(buff.p[a])
    @. buff.p *= -one(logpa)
    buff.p[a] += one(logpa)
    G = buff.ψ.weight'
    for i in 1:length(buff.p)
        @. G[i, :] = s * buff.p[i]
    end
    return logpa, buff.ψ
end

function grad_logpdf_linear_softmax(ψ, θ, s::AbstractMatrix, a)
    T = eltype(θ)
    num_a = size(θ,2)
    n = size(s,2)
    p = zeros(T, (num_a, n))
    mul!(p, θ', s)
    logps = softmax_batch_out!(p, a)

    dp = p
    @. dp *= -one(T)
    for i in 1:n
        dp[a[i], i] += one(T)
    end
    
    mul!(ψ, s, dp')
    return logps
end

function grad_logpdf(layer::LinearSoftmax, x, a, params, states)
    buff = make_buff(layer, params)
    logpa, ψ = grad_logpdf_linear_softmax!(buff, params.weight, x, a)
    return logpa, ψ, states
end

function grad_logpdf(layer::LinearSoftmax{Val{true}}, x, a, params, states)
    logpa, ψ = grad_logpdf_linear_softmax!(states, params.weight, x, a)
    return logpa, ψ, states 
end

function grad_logpdf(layer::LinearSoftmax, x::AbstractMatrix, a, params, states)
    ψ = zero(params)
    logpa = grad_logpdf_linear_softmax(ψ.weight, params.weight, x, a)
    return logpa, ψ, states
end

function grad_logpdf(layer::LinearSoftmax{Val{true}}, x::AbstractMatrix, a, params, states)
    ψ = states.ψ
    logpa = grad_logpdf_linear_softmax(ψ.weight, params.weight, x, a)
    return logpa, ψ, states
end


function sample_with_trace!(buff, θ, s::AbstractVector)
    mul!(buff.p, θ', s)
    softmax!(buff.p)
    a = sample_discrete(buff.p)
    buff.action[1] = a
    logp = log(buff.p[a])
    @. buff.p *= -one(logp)
    buff.p[a] += one(logp)
    G = buff.ψ.weight'
    for i in 1:length(buff.p)
        @. G[i, :] = s * buff.p[i]
    end
    return a, logp, buff.ψ
end

function sample_with_trace(layer::LinearSoftmax, s::AbstractVector, params, states)
    buff = make_buff(layer, params)
    action, logp, ψ = sample_with_trace!(buff, params.weight, s)
    return action, logp, ψ, states
end

function sample_with_trace(layer::LinearSoftmax{Val{true}}, s::AbstractVector, params, states)
    action, logp, ψ = sample_with_trace!(states, params.weight, s)
    return action, logp, ψ, states
end
