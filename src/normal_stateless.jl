
struct StatelessNormal{B,F1,F2} <: LuxCore.AbstractExplicitLayer
    num_actions::Int
    init_mean::F1
    init_std::F2
end

function StatelessNormal(num_actions::Int; init_mean=zeros32, init_std=zeros32, buffered::Bool=false)
    return StatelessNormal{typeof(Val(buffered)), typeof(init_mean), typeof(init_std)}(num_actions, init_mean, init_std)
end

function LuxCore.initialparameters(rng::Random.AbstractRNG, layer::StatelessNormal)
    μ = layer.init_mean(rng, layer.num_actions)
    logσ = layer.init_std(rng, layer.num_actions)
    if layer.num_actions == 1
        return (; μ=μ[1], logσ=logσ[1])
    else
        return (; μ=μ, logσ=logσ)
    end
end

LuxCore.initialstates(rng::Random.AbstractRNG, layer::StatelessNormal{Val{false}}) = NamedTuple()

function LuxCore.initialstates(rng::Random.AbstractRNG, layer::StatelessNormal{Val{true}})
    T = eltype(layer.init_mean(rng, 1,1))
    if layer.num_actions == 1
        action = zero(T)
    else
        action = zeros(T, layer.num_actions)
    end
    σ = zero(action)
    ψ = (;μ = zero(action), logσ=zero(action)) |> ComponentArray
    return (;action=action, σ=σ, ψ=ψ)
end

LuxCore.parameterlength(layer::StatelessNormal) = layer.num_actions * 2

LuxCore.statelength(layer::StatelessNormal) = 0
LuxCore.statelength(layer::StatelessNormal{Val{true}}) = layer.num_actions * 4


function comp_sigma(logσ::AbstractVector, states::NamedTuple{(), Tuple{}})
    σ = @. exp(logσ)
    return σ
end

function comp_sigma(logσ::Real, states)
    σ = exp(logσ)
    return σ
end

function comp_sigma(logσ::AbstractVector, states::Union{NamedTuple{(:action, :σ, :ψ)},NamedTuple{(:action, :μ, :σ, :dμ, :ψ)}})
    @. states.σ = exp(logσ)
    return states.σ
end

function rrule(::typeof(comp_sigma), logσ::AbstractVector, states::Union{NamedTuple{(:action, :σ, :ψ)},NamedTuple{(:action, :μ, :σ, :dμ, :ψ)}})
    @. states.σ = exp(logσ)
    function comp_sigma_pullback(ȳ)
        dσ = @thunk begin
            dp = states.ψ.logσ
            @. dp = ȳ * states.σ
            return dp
        end
        return NoTangent(), dσ, NoTangent()
    end
    return states.σ, comp_sigma_pullback
end

function make_normal_dist(μ::T, σ::T) where {T <: Real}
    return Normal(μ, σ)
end

function make_normal_dist(μ::AbstractVector{T}, σ::AbstractVector{T}) where {T <: Real}
    return MvNormal(μ, Diagonal(map(abs2, σ)))
end


function (layer::StatelessNormal)(params, states)
    σ = comp_sigma(params.logσ, states)
    μ = params.μ
    d = make_normal_dist(μ, σ)
    return d, states
end

function (layer::StatelessNormal)(x, params, states)
    return layer(params, states)
end


function logpdf_normal(z::T, μ::T, σ::T) where {T <:Real}
    tmp = -log(sqrt(T(2.0) * T(π)))
    logp = tmp - (z - μ)^2 / (T(2.0) * σ^2) - log(σ)
    return logp
end

function logpdf_normal(z::Real, μ::T, σ::T) where {T <:Real}
    return logpdf_normal(convert(T, z), μ, σ)
end


function logpdf_normal(z::AbstractVector, μ::AbstractVector{T}, σ::AbstractVector) where {T}
    logp = zero(T)
    for i in eachindex(z)
        logp += logpdf_normal(z[i], μ[i], σ[i])
    end
    return logp
end


function logpdf(layer::StatelessNormal, a, params, states)
    σ = comp_sigma(params.logσ, states)
    μ = params.μ
    logp = logpdf_normal(a, μ, σ)
    return logp, states
end

function select_action_batch(A, i, μ::Real)
    return A[1,i]
end

function select_action_batch(A, i, μ::AbstractVector)
    return @view A[:, i]
end

function logpdf(layer::StatelessNormal, a::AbstractMatrix{T}, params, states) where {T}
    σ = comp_sigma(params.logσ, states)
    μ = params.μ
    logps = [logpdf_normal(select_action_batch(a, i, μ), μ, σ) for i in axes(a,2)]
    return logps, states
end

function logpdf(layer::StatelessNormal, x, a, params, states)
    return logpdf(layer, a, params, states)
end
function grad_normal_logpdf(dμ::T, dlogσ::T, z::Real, μ::T, σ::T) where {T <:Real}
    return grad_normal_logpdf(dμ, dlogσ, convert(T, z), μ, σ)
end
function grad_normal_logpdf(dμ::T, dlogσ::T, z::T, μ::T, σ::T) where {T <:Real}
    zmμ = z - μ
    σ² = σ^2
    dμ = zmμ / σ²
    dσ = (-one(T) + zmμ^2 / σ²) # / σ # (σ cancels out) for logσ
    tmp = -log(sqrt(T(2.0) * T(π)))
    logp = tmp - zmμ^2 / (T(2.0) * σ²) - log(σ)
    return logp, dμ, dσ
end

function grad_normal_logpdf(dμ, dlogσ, z::AbstractVector, μ::AbstractVector{T}, σ::AbstractVector) where {T}
    tmp = -log(sqrt(T(2.0) * T(π)))
    logp = zero(T)
    for i in eachindex(μ)
        zmμ = z[i] - μ[i]
        σ² = σ[i]^2
        logp += tmp - zmμ^2 / (T(2.0) * σ²) - log(σ[i])
        dμ[i] += zmμ / σ²
        dlogσ[i] += (-one(T) + zmμ^2 / σ²) #/ σ[i] * σ[i] # (σ[i] * σ[i] cancels out)
    end
    return logp, dμ, dlogσ
end



function grad_logpdf(layer::StatelessNormal, a, params, states)
    σ = comp_sigma(params.logσ, states)
    μ = params.μ
    dμ = zero(μ)
    dlogσ = zero(σ)
    logp, dμ, dlogσ = grad_normal_logpdf(dμ, dlogσ, a, μ, σ)
    dparams = (;μ=dμ, logσ=dlogσ) |> ComponentArray
    return logp, dparams, states
end

function stateless_grad(ψ, dμ::T, dlogσ::T) where {T <: Real}
    ψ.μ = dμ
    ψ.logσ = dlogσ
    return nothing    
end

function stateless_grad(ψ, dμ::AbstractVector, dlogσ::AbstractVector) 
    return nothing    
end

function grad_logpdf(layer::StatelessNormal{Val{true}}, a, params, states)
    σ = comp_sigma(params.logσ, states)
    μ = params.μ
    T = eltype(μ)
    fill!(states.ψ, zero(T))
    logp, dμ, dlogσ = grad_normal_logpdf(states.ψ.μ, states.ψ.logσ, a, μ, σ)
    stateless_grad(states.ψ, dμ, dlogσ)
    return logp, states.ψ, states
end

function grad_normal_stateless_batch(ψ::ComponentArray, μ::T, σ::T, A) where {T}
    num_obs = size(A, 2)
    logps = zeros(T, num_obs)
    fill!(ψ, zero(T))
    for i in axes(A, 2)
        ai = A[1, i]  # only one dim so must be a single action
        logp, dμ, dlogσ = grad_normal_logpdf(ψ.μ, ψ.logσ, ai, μ, σ)
        ψ.μ += dμ
        ψ.logσ += dlogσ
        logps[i] = logp
    end
    return logps, ψ
end

function grad_normal_stateless_batch(ψ::ComponentArray, μ::AbstractVector{T}, σ::AbstractVector{T}, A::AbstractMatrix) where {T}
    num_obs = size(A, 2)
    logps = zeros(T, num_obs)
    fill!(ψ, zero(T))
    for i in axes(A, 2)
        ai = @view A[:, i]  
        logp, dμ, dlogσ = grad_normal_logpdf(ψ.μ, ψ.logσ, ai, μ, σ)  # grads for μ and logσ will be accumulated
        logps[i] = logp
    end
    return logps, ψ
end

function grad_logpdf(layer::StatelessNormal, a::AbstractMatrix{T}, params, states) where {T}
    σ = comp_sigma(params.logσ, states)
    μ = params.μ
    ψ = (μ = zero(μ), logσ = zero(σ)) |> ComponentArray
    logps, ψ = grad_normal_stateless_batch(ψ, μ, σ, a)
    return logps, ψ, states
end

function grad_logpdf(layer::StatelessNormal{Val{true}}, a::AbstractMatrix{T}, params, states) where {T}
    σ = comp_sigma(params.logσ, states)
    μ = params.μ
    logps, ψ = grad_normal_stateless_batch(states.ψ, μ, σ, a)
    return logps, ψ, states
end

function grad_logpdf(layer::StatelessNormal, x, a, params, states)
    return grad_logpdf(layer, a, params, states)
end

function sample_normal(a, μ::T, σ::T) where {T <: Real}
    return μ + σ * randn(T)
end

function sample_normal(a::AbstractVector{T}, μ::AbstractVector{T}, σ::AbstractVector{T}) where {T}
    @. a = μ + σ * randn((T,))
    return a
end


function sample_with_trace(layer::StatelessNormal, params, states)
    σ = comp_sigma(params.logσ, states)
    μ = params.μ
    T = eltype(μ)
    action = zero(μ)
    action = sample_normal(action, μ, σ)
    ψ = (;μ=zero(μ), logσ=zero(σ)) |> ComponentArray
    logp, dμ, dlogσ = grad_normal_logpdf(ψ.μ, ψ.logσ, action, μ, σ)
    stateless_grad(ψ, dμ, dlogσ)
    return action, logp, ψ, states
end

function sample_with_trace(layer::StatelessNormal{Val{true}}, params, states)
    σ = comp_sigma(params.logσ, states)
    μ = params.μ
    T = eltype(μ)
    action = states.action
    action = sample_normal(action, μ, σ)
    ψ = states.ψ
    fill!(ψ, zero(T))
    logp, dμ, dlogσ = grad_normal_logpdf(ψ.μ, ψ.logσ, action, μ, σ)
    stateless_grad(ψ, dμ, dlogσ)
    return action, logp, ψ, states
end

function sample_with_trace(layer::StatelessNormal, x, params, states)
    return sample_with_trace(layer, params, states)
end