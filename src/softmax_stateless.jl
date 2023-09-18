
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

@non_differentiable sample_discrete(p::Any)

struct StatelessSoftmax{B,F1} <: LuxCore.AbstractExplicitLayer
    num_actions::Int
    init_weights::F1
end

function StatelessSoftmax(num_actions::Int; init_weights=zeros32, buffered::Bool=false)
    return StatelessSoftmax{typeof(Val(buffered)), typeof(init_weights)}(num_actions, init_weights)
end

function LuxCore.initialparameters(rng::Random.AbstractRNG, layer::StatelessSoftmax)
    return (; weight=layer.init_weights(rng, layer.num_actions))
end

LuxCore.initialstates(rng::Random.AbstractRNG, layer::StatelessSoftmax) = NamedTuple()

function LuxCore.initialstates(rng::Random.AbstractRNG, layer::StatelessSoftmax{Val{true}})
    T = eltype(layer.init_weights(rng, 1,1))
    action = zeros(Int, 1)
    p = zeros(T, layer.num_actions)
    ψ = (;weight = zeros(T, layer.num_actions)) |> ComponentArray
    
    return (;action=action, p=p, ψ=ψ)
end

LuxCore.parameterlength(layer::StatelessSoftmax) = layer.num_actions

LuxCore.statelength(layer::StatelessSoftmax) = 0
LuxCore.statelength(layer::StatelessSoftmax{Val{true}}) = layer.num_actions + layer.num_actions + 1

function (layer::StatelessSoftmax)(params, states)
    p = softmax(params.weight)
    return Categorical(p), states
end

function (layer::StatelessSoftmax{Val{true}})(params, states)
    buffer = states.p
    softmax!(buffer, params.weight)
    return Categorical(buffer), states
end

function (layer::StatelessSoftmax)(x, params, states)
    return layer(params, states)
end

function (layer::StatelessSoftmax)(x::AbstractMatrix, params, states)
    p = softmax(params.weight)
    num_obs = size(x, 2)
    pall = repeat(p, inner=(1, num_obs))
    return pall, states
end


function make_buff(layer::StatelessSoftmax, params)
    action = zeros(Int, 1)
    p = zero(params.weight)
    dW = zero(params.weight)
    ψ = (;weight=dW) |> ComponentArray
    buff = (action=action, p=p, ψ=ψ)
    return buff
end

@non_differentiable make_buff(layer::StatelessSoftmax, params::Any)

function logpdf(layer::StatelessSoftmax, a, params, states)
    return logpdf_stateless_softmax(params.weight, a), states
end

function logpdf(layer::StatelessSoftmax{Val{true}}, a::Int, params, states)
    return logpdf_stateless_softmax(states, params.weight, a), states
end

function logpdf(layer::StatelessSoftmax, x, a, params, states)
    return logpdf(layer, a, params, states)
end

function logpdf_stateless_softmax(buff, θ, a::Int)
    softmax!(buff.p, θ)
    logp = log(buff.p[a])
    return logp
end

function grad_logpdf_stateless_softmax!(dp::AbstractVector, p::AbstractVector, a::Int)
    @. dp = -p
    dp[a] += 1
    return dp
end

function rrule(::typeof(logpdf_stateless_softmax), buff, θ, a::Int)
    T = eltype(θ)
    softmax!(buff.p, θ)
    logp = log(buff.p[a])

    function logpdf_softmax_pullback(ȳ)
        dθ = @thunk begin
            ψ = buff.ψ.weight
            grad_logpdf_stateless_softmax!(ψ, buff.p, a)
            @. ψ *= ȳ
            return ψ
        end
        return NoTangent(), NoTangent(), dθ, NoTangent()
    end
    return logp, logpdf_softmax_pullback
end

function logpdf_stateless_softmax(θ, a::Int)
    p = softmax(θ)
    logp = log(p[a])
    return logp
end

function rrule(::typeof(logpdf_stateless_softmax), θ, a::Int)
    T = eltype(θ)
    p = softmax(θ)
    logp = log(p[a])

    function logpdf_stateless_softmax_pullback(ȳ)
        dθ = @thunk begin 
            dp = zero(p)
            grad_logpdf_stateless_softmax!(dp, p, a)
            @. dp *= ȳ
        end
        return NoTangent(), dθ, NoTangent()
    end
    return logp, logpdf_stateless_softmax_pullback
end

function logpdf_stateless_softmax(θ, a::AbstractVector)
    p = softmax(θ)
    logps = zeros(eltype(θ), length(a))
    for i in eachindex(a)
        logps[i] = log(p[a[i]])
    end
    return logps
end

function rrule(::typeof(logpdf_stateless_softmax), θ, a::AbstractVector)
    T = eltype(θ)
    p = softmax(θ)
    logps = zeros(T, length(a))
    for i in eachindex(a)
        logps[i] = log(p[a[i]])
    end

    function logpdf_stateless_softmax_batch_pullback(ȳ)
        dθ = @thunk begin
            dp = -p * sum(ȳ)
            for i in eachindex(a)
                dp[a[i]] += ȳ[i]
            end
            return dp
        end
        return NoTangent(), dθ, NoTangent()
    end
    return logps, logpdf_stateless_softmax_batch_pullback
end



function grad_logpdf(layer::StatelessSoftmax, a::Int, params, states)
    buff = make_buff(layer, params)
    softmax!(buff.p, params.weight)
    logpa = buff.p[a]
    grad_logpdf_stateless_softmax!(buff.ψ.weight, buff.p, a)
    return logpa, buff.ψ, states
end

function grad_logpdf(layer::StatelessSoftmax{Val{true}}, a::Int, params, states)
    softmax!(states.p, params.weight)
    logpa = states.p[a]
    grad_logpdf_stateless_softmax!(states.ψ.weight, states.p, a)
    return logpa, states.ψ, states
end

function grad_logpdf_stateless_softmax!(dp::AbstractVector, p::AbstractVector, a::AbstractVector)
    n = length(a)
    @. dp = -p * n
    for i in eachindex(a)
        dp[a[i]] += 1
    end
    return dp
end

function grad_logpdf(layer::StatelessSoftmax, a::AbstractVector{Int}, params, states)
    ψ = zero(params)
    p = softmax(params.weight)
    logpa = zeros(eltype(p), length(a))
    for i in eachindex(a)
        logpa[i] = log(p[a[i]])
    end
    grad_logpdf_stateless_softmax!(ψ.weight,p, a)
    return logpa, ψ, states
end

function grad_logpdf(layer::StatelessSoftmax{Val{true}}, a::AbstractVector{Int}, params, states)
    ψ = states.ψ
    softmax!(states.p, params.weight)
    logpa = zeros(eltype(states.p), length(a))
    for i in eachindex(a)
        logpa[i] = log(states.p[a[i]])
    end
    grad_logpdf_stateless_softmax!(ψ.weight, states.p, a)
    return logpa, ψ, states
end

function grad_logpdf(layer::StatelessSoftmax, x, a, params, states)
    return grad_logpdf(layer, a, params, states)
end

function sample_with_trace!(buff, θ)
    softmax!(buff.p, θ)
    a = sample_discrete(buff.p)
    buff.action[1] = a
    logp = log(buff.p[a])
    grad_logpdf_stateless_softmax!(buff.ψ.weight, buff.p, a)
    return a, logp, buff.ψ
end

function sample_with_trace(layer::StatelessSoftmax, params, states)
    buff = make_buff(layer, params)
    action, logp, ψ = sample_with_trace!(buff, params.weight)
    return action, logp, ψ, states
end

function sample_with_trace(layer::StatelessSoftmax{Val{true}}, params, states)
    action, logp, ψ = sample_with_trace!(states, params.weight)
    return action, logp, ψ, states
end

function sample_with_trace(layer::StatelessSoftmax, x, params, states)
    return sample_with_trace(layer, params, states)
end