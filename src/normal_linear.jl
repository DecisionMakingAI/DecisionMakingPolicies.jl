
struct LinearNormal{B,F1,F2} <: LuxCore.AbstractExplicitLayer
    in_dims::Int
    num_actions::Int
    init_weights::F1
    init_std::F2
end

function LinearNormal(in_dims::Int, num_actions::Int; init_weights=zeros32, init_std=zeros32, buffered::Bool=false)
    return LinearNormal{typeof(Val(buffered)), typeof(init_weights), typeof(init_std)}(in_dims, num_actions, init_weights, init_std)
end


function LuxCore.initialparameters(rng::Random.AbstractRNG, layer::LinearNormal)
    weight=layer.init_weights(rng, layer.in_dims, layer.num_actions)
    logσ=layer.init_std(rng, layer.num_actions)
    if layer.num_actions == 1
        return (; weight=weight[:, 1], logσ=logσ[1])
    else
        return (; weight=weight, logσ=logσ)
    end
end

LuxCore.initialstates(rng::Random.AbstractRNG, layer::LinearNormal) = NamedTuple()

function LuxCore.initialstates(rng::Random.AbstractRNG, layer::LinearNormal{Val{true}})
    T = eltype(layer.init_weights(rng, 1,1))
    if layer.num_actions == 1
        action = zero(T)
        weight = zeros(T, layer.in_dims)
    else
        action = zeros(T, layer.num_actions)
        weight = zeros(T, layer.in_dims, layer.num_actions)
    end
    μ = zero(action)
    dμ = zero(action)
    σ = zero(action)
    ψ = (;weight = weight, logσ=zero(σ)) |> ComponentArray
    
    return (;action=action, μ=μ, σ=σ, dμ=dμ, ψ=ψ)
end

LuxCore.parameterlength(layer::LinearNormal) = layer.in_dims * layer.num_actions + layer.num_actions

LuxCore.statelength(layer::LinearNormal) = 0
LuxCore.statelength(layer::LinearNormal{Val{true}}) = layer.in_dims * layer.num_actions + 5 * layer.num_actions


function compute_mu(W::AbstractMatrix{T}, x::AbstractVector) where {T}
    μ = zeros(T, size(W, 2))
    μ = compute_mu!(μ, W, x)
    return μ
end

function compute_mu(W::AbstractMatrix{T}, x::AbstractMatrix) where {T}
    μ = zeros(T, (size(W, 2), size(x, 2)))
    μ = compute_mu!(μ, W, x)
    return μ
end

function compute_mu(W::AbstractVector{T}, x::AbstractMatrix{T}) where {T}
    μ = W' * x
    return μ
end

function compute_mu(W::AbstractVector{T}, x::AbstractVector) where {T}
    return convert(T, dot(W, x))
end

function compute_mu!(μ::AbstractArray, W, x)
    mul!(μ, W', x)
    return μ
end

function rrule(::typeof(compute_mu!), μ::AbstractArray, W, x)
    mul!(μ, W', x)
    function compute_mu_pullback(ȳ)
        dW = @thunk begin
            d = zero(W)
            mul!(d, x, ȳ')
            return d
        end
        dx = @thunk begin 
            d = zero(x)
            mul!(d, W, ȳ)
            return d
        end
        return NoTangent(), NoTangent(), dW, dx
    end
    return μ, compute_mu_pullback
end

function compute_mu!(μ::Real, W::AbstractVector{T}, x::AbstractVector) where {T}
    μ = convert(T, dot(W, x))
    return μ
end
    
function (layer::LinearNormal)(x, params, states)
    μ = compute_mu(params.weight, x)
    σ = comp_sigma(params.logσ, states)
    d = make_normal_dist(μ, σ)
    return d, states
end

function (layer::LinearNormal{Val{true}})(x, params, states)
    μ = compute_mu!(states.μ, params.weight, x)
    σ = comp_sigma(params.logσ, states)
    d = make_normal_dist(μ, σ)
    return d, states
end


function logpdf(layer::LinearNormal, x, a, params, states)
    μ = compute_mu(params.weight, x)
    σ = comp_sigma(params.logσ, states)
    logp = logpdf_normal(a, μ, σ)
    return logp, states
end

function logpdf(layer::LinearNormal{Val{true}}, x, a, params, states)
    μ = compute_mu!(states.μ, params.weight, x)
    σ = comp_sigma(params.logσ, states)
    logp = logpdf_normal(a, μ, σ)
    return logp, states
end

function logpdf_normal_batch(z::AbstractMatrix, μ::AbstractMatrix{T}, σ, bidx::Int) where {T}
    logp = zero(T)
    zb = select_action_batch(z, bidx, σ)
    μb = select_action_batch(μ, bidx, σ)
    for i in eachindex(σ)
        logp += logpdf_normal(zb[i], μb[i], σ[i])
    end
    return logp
end

function normal_batch_logpdf(x, a, W, σ)
    μ = compute_mu(W, x)
    logps = [logpdf_normal_batch(a, μ, σ, i) for i in axes(a,2)]
    return logps
end
function logpdf(layer::LinearNormal, x::AbstractMatrix, a::AbstractMatrix{T}, params, states) where {T}
    x = type_check(x, params.weight)
    σ = comp_sigma(params.logσ, states)
    logps = normal_batch_logpdf(x, a, params.weight, σ)
    return logps, states
end

function logpdf(layer::LinearNormal{Val{true}}, x::AbstractMatrix, a::AbstractMatrix{T}, params, states) where {T}
    x = type_check(x, params.weight)
    σ = comp_sigma(params.logσ, states)
    logps = normal_batch_logpdf(x, a, params.weight, σ)
    return logps, states
end

 function grad_buff(params::ComponentArray)
    return zero(params)
end

function grad_buff(params::NamedTuple)
    g = ComponentArray(params)
    fill!(g, zero(eltype(g)))
    return g
end

function grad_buff(params, states)
    dμ = zero(params.logσ)
    g = grad_buff(params)
    return g, dμ
end

function grad_buff(params, states::NamedTuple{(:action, :μ, :σ, :dμ, :ψ)})
    dμ = states.dμ
    g = states.ψ
    T = eltype(g)
    fill!(g, zero(T))
    if length(states.action) == 1
        dμ = zero(T)
    else
        fill!(dμ, zero(T))
    end
    return g, dμ
end

function dlogσ_grad(ψ, dlogσ::T) where {T <: Real}
    ψ.logσ = dlogσ
    return nothing    
end

function dlogσ_grad(ψ, dlogσ::AbstractVector) 
    return nothing    
end

function grad_linear(x::AbstractVector{T}, dy::AbstractVector{T}, dW::AbstractMatrix{T}) where {T}
    BLAS.ger!(one(T), x, dy, dW)
    return nothing
end

function grad_linear(x::AbstractMatrix{T}, dy::AbstractMatrix{T}, dW::AbstractArray{T}) where {T}
    mul!(dW, x, dy')
    return nothing
end

function grad_linear(x::AbstractVector, dy::Real, dW::AbstractVector)
    @. dW = x * dy
    return nothing
end

function type_check(x::AbstractArray{T}, W::AbstractArray{T}) where {T}
    return x
end

function type_check(x::AbstractArray{T}, W::AbstractArray{T2}) where {T,T2}
    y = convert.(T2, x)
    return y
end

function grad_logpdf(layer::LinearNormal, x, a, params, states)
    x = type_check(x, params.weight)
    μ = compute_mu(params.weight, x)
    σ = comp_sigma(params.logσ, states)
    g, dμ = grad_buff(params, states)
    fill!(g, zero(eltype(g)))
    dlogσ = g.logσ
    logp, dμ, dlogσ = grad_normal_logpdf(dμ, dlogσ, a, μ, σ)
    dlogσ_grad(g, dlogσ)
    grad_linear(x, dμ, g.weight)
    return logp, g, states
end



function grad_normal_linear_batch(ψ::ComponentArray, W::AbstractArray{T}, σ::AbstractVector{T}, x::AbstractMatrix{T}, A::AbstractMatrix) where {T}
    num_obs = size(A, 2)
    logps = zeros(T, num_obs)
    fill!(ψ, zero(T))
    μ = compute_mu(W, x)
    dμ = zero(μ)
    for i in axes(A, 2)
        ai = @view A[:, i]  
        μi = @view μ[:, i]
        dμi = @view dμ[:, i]
        logp, dμi, dlogσ = grad_normal_logpdf(dμi, ψ.logσ, ai, μi, σ)  # grads for μ and logσ will be accumulated
        logps[i] = logp
    end
    grad_linear(x, dμ, ψ.weight)
    return logps, ψ
end

function grad_normal_linear_batch(ψ::ComponentArray, W::AbstractArray{T}, σ::T, x::AbstractMatrix{T}, A::AbstractMatrix) where {T}
    num_obs = size(A, 2)
    logps = zeros(T, num_obs)
    fill!(ψ, zero(T))
    μ = compute_mu(W, x)
    dμ = zero(μ)
    for i in axes(A, 2)
        ai = A[1, i]  
        μi = μ[1, i]
        dμi = dμ[1, i]
        logp, dμi, dlogσ = grad_normal_logpdf(zero(T), zero(T), ai, μi, σ)
        dμ[1, i] += dμi
        ψ.logσ += dlogσ
        logps[i] = logp
    end
    grad_linear(x, dμ, ψ.weight)
    return logps, ψ
end

function grad_logpdf(layer::LinearNormal, x::AbstractMatrix, a::AbstractMatrix{T}, params, states) where {T}
    x = type_check(x, params.weight)
    σ = comp_sigma(params.logσ, states)
    g = grad_buff(params)
    logps, ψ = grad_normal_linear_batch(g, params.weight, σ, x, a)
    return logps, ψ, states
end

function grad_logpdf(layer::LinearNormal{Val{true}}, x::AbstractMatrix, a::AbstractMatrix{T}, params, states) where {T}
    x = type_check(x, params.weight)
    σ = comp_sigma(params.logσ, states)
    logps, ψ = grad_normal_linear_batch(states.ψ, params.weight, σ, x, a)
    return logps, ψ, states
end

function sample_with_trace(layer::LinearNormal, x, params, states)
    x = type_check(x, params.weight)
    μ = compute_mu(params.weight, x)
    σ = comp_sigma(params.logσ, states)
    action = zero(μ)
    action = sample_normal(action, μ, σ)
    ψ, dμ = grad_buff(params, states)
    logp, dμ, dlogσ = grad_normal_logpdf(dμ, ψ.logσ, action, μ, σ)
    dlogσ_grad(ψ, dlogσ)
    grad_linear(x, dμ, ψ.weight)
    return action, logp, ψ, states
end

function sample_with_trace(layer::LinearNormal{Val{true}}, x, params, states)
    x = type_check(x, params.weight)
    μ = compute_mu(params.weight, x)
    σ = comp_sigma(params.logσ, states)
    T = eltype(μ)
    action = states.action
    action = sample_normal(action, μ, σ)
    ψ, dμ = grad_buff(params, states)
    logp, dμ, dlogσ = grad_normal_logpdf(dμ, ψ.logσ, action, μ, σ)
    dlogσ_grad(ψ, dlogσ)
    grad_linear(x, dμ, ψ.weight)
    return action, logp, ψ, states
end