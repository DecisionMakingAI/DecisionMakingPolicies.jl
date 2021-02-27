using DecisionMakingPolicies
using Test

import Zygote: gradient
import Distributions: Normal, MvNormal

@testset "Stateless Normal Tests" begin
    num_actions = 1
    T = Float32
    p = StatelessNormal()
    θ = initparams(p, T, num_actions)
    @test eltype(θ[1]) == T
    @test eltype(θ[2]) == T
    @test size(θ[1]) == (num_actions,)
    @test size(θ[2]) == (num_actions,)
    d = p(θ)
    a = T(1.0)
    logp1 = logpdf(d, a)
    @test typeof(d) <: Normal{T}
    @test typeof(logp1) == T

    num_actions = 2
    θ = initparams(p, T, num_actions)
    @test typeof(p(θ)) <: MvNormal

    a = rand(p(θ))
    @test eltype(a) == T
    @test size(a) == (2,)
    a = Array{T,1}([1.0, 0.0])
    d = p(θ)
    logp1 = logpdf(d, a)
    logp2 = logpdf(p, θ, a)
    @test typeof(logp2) == T
    @test_broken typeof(logp2) == typeof(logp1)
    @test isapprox(logp1, logp2)
    g = (similar(θ[1]), similar(θ[2]))
    gauto1 = gradient(θ->logpdf(p(θ),a),θ)[1]
    gauto2 = gradient(θ->logpdf(p,θ,a),θ)[1]

    @test_broken eltype(gauto1[1]) == eltype(gauto2[1])
    @test_broken eltype(gauto1[2]) == eltype(gauto2[2])
    @test eltype(gauto2[1]) == eltype(θ[1])
    @test eltype(gauto2[2]) == eltype(θ[2])
    @test all(isapprox.(gauto1[1], gauto2[1]))
    @test all(isapprox.(gauto1[2], gauto2[2]))

    logp3 = grad_logpdf!(g, p, θ, a)
    @test typeof(logp3) == T
    @test logp3 == logp2
    @test all(isapprox.(g[1], gauto1[1]))
    @test all(isapprox.(g[2], gauto1[2]))
    g2 = (Array{T,1}([1.0, 0.0]), Array{T,1}([0.0, -1.0]))

    @test all(isapprox.(g[1], g2[1]))
    @test all(isapprox.(g[2], g2[2]))

    A = zeros(T,2)
    A2 = zeros(T,(2,2))
    logp4 = sample_with_trace!(g, A, p, θ)
    @test all(.!isapprox.(A, 0.0))
    @test all(isapprox.(g[1], A))
    @test all(isapprox.(g[2], A.^2 .- 1))

    sample_with_trace!(g, view(A2,:,1), p, θ)
    @test all(.!isapprox.(A2[:,1], 0.0))
    @test all(isapprox.(A2[:,2], 0.0))
    a1 = A2[:,1]
    sample_with_trace!(g, view(A2,:, 2), p, θ)
    @test all(isapprox.(A2[:, 1], a1))
    @test all(.!isapprox.(A2[:,2], 0.0))
end

@testset "Stateless Normal Buffer Tests" begin
    num_actions = 2
    T = Float32
    p = StatelessNormal()
    θ = initparams(p, T, num_actions)
    buff = NormalBuffer(p, θ)
    @test eltype(buff.μ) == T
    @test eltype(buff.action) == T
    @test eltype(buff.ψ[1]) == T
    @test eltype(buff.ψ[2]) == T
    @test size(buff.μ) == (num_actions,)
    @test size(buff.action) == (num_actions,)
    @test size(buff.ψ[1]) == (num_actions,)
    @test size(buff.ψ[2]) == (num_actions,)
    d = p(buff, θ)
    @test eltype(d.μ) == T
    a = Array{T,1}([1.0, 0.0])
    logp1 = logpdf!(buff, p, θ, a)
    @test typeof(logp1) == T

    
    
    g = (similar(θ[1]), similar(θ[2]))
    gauto1 = gradient(θ->logpdf(p,θ,a),θ)[1]
    logp3 = grad_logpdf!(buff, p, θ, a)
    @test typeof(logp3) == T
    @test logp3 == logp1
    @test all(isapprox.(buff.ψ[1], gauto1[1]))
    @test all(isapprox.(buff.ψ[2], gauto1[2]))
    g2 = (Array{T,1}([1.0, 0.0]), Array{T,1}([0.0, -1.0]))

    @test all(isapprox.(buff.ψ[1], g2[1]))
    @test all(isapprox.(buff.ψ[2], g2[2]))

    logp4 = sample_with_trace!(buff, p, θ)
    @test typeof(logp4) == T
    @test all(.!isapprox.(buff.action, 0.0))
    @test all(isapprox.(buff.ψ[1], buff.action))
    @test all(isapprox.(buff.ψ[2], buff.action.^2 .- 1))
end


@testset "Linear Normal Tests" begin
    num_actions = 1
    num_features = 3
    T = Float32
    p = LinearNormal()
    θ = initparams(p, T, num_features, num_actions)
    s = Array{T, 1}([0.0, 1.0, 2.0])

    @test eltype(θ[1]) == T
    @test eltype(θ[2]) == T
    @test size(θ[1]) == (num_features, num_actions)
    @test size(θ[2]) == (num_actions,)
    @test typeof(p(θ,s)) <: Normal

    num_actions = 2
    θ = initparams(p, T, num_features, num_actions)
    @test typeof(p(θ,s)) <: MvNormal

    a = rand(p(θ,s))
    @test eltype(a) == T
    @test size(a) == (2,)
    a = Array{T,1}([1.0, 0.0])

    logp1 = logpdf(p(θ,s), a)
    logp2 = logpdf(p, θ, s, a)

    @test typeof(logp2) == T
    @test_broken typeof(logp2) == typeof(logp1)
    @test isapprox(logp1, logp2)
    g = (similar(θ[1]), similar(θ[2]))
    gauto1 = gradient(θ->logpdf(p(θ,s),a),θ)[1]
    gauto2 = gradient(θ->logpdf(p,θ,s,a),θ)[1]

    @test_broken eltype(gauto1[1]) == eltype(gauto2[1])
    @test_broken eltype(gauto1[2]) == eltype(gauto2[2])
    @test eltype(gauto2[1]) == eltype(θ[1])
    @test eltype(gauto2[2]) == eltype(θ[2])
    @test all(isapprox.(gauto1[1], gauto2[1]))
    @test all(isapprox.(gauto1[2], gauto2[2]))

    logp3 = grad_logpdf!(g, p, θ, s, a)
    @test typeof(logp3) == T
    @test logp3 == logp2
    @test all(isapprox.(g[1], gauto1[1]))
    @test all(isapprox.(g[2], gauto1[2]))
    g2 = (Array{T,2}([0.0 0.0; 1.0 0.0; 2.0 0.0]), Array{T,1}([0.0, -1.0]))

    @test all(isapprox.(g[1], g2[1]))
    @test all(isapprox.(g[2], g2[2]))

    A = zeros(T,2)
    A2 = zeros(T,(2,2))
    logp4 = sample_with_trace!(g, A, p, θ, s)
    @test all(.!isapprox.(A, 0.0))
    @test all(isapprox.(g[1], s * A'))
    @test all(isapprox.(g[2], A.^2 .- 1))

    sample_with_trace!(g, view(A2,:,1), p, θ, s)
    @test all(.!isapprox.(A2[:,1], 0.0))
    @test all(isapprox.(A2[:,2], 0.0))
    a1 = A2[:,1]
    sample_with_trace!(g, view(A2,:, 2), p, θ, s)
    @test all(isapprox.(A2[:, 1], a1))
    @test all(.!isapprox.(A2[:,2], 0.0))
end


@testset "Linear Normal Buffer Tests" begin
    num_actions = 2
    num_features = 3
    T = Float32
    p = LinearNormal()
    θ = initparams(p, T, num_features, num_actions)
    s = Array{T, 1}([0.0, 1.0, 2.0])
    buff = NormalBuffer(p, θ)
    @test eltype(buff.μ) == T
    @test eltype(buff.action) == T
    @test eltype(buff.ψ[1]) == T
    @test eltype(buff.ψ[2]) == T
    @test size(buff.μ) == (num_actions,)
    @test size(buff.action) == (num_actions,)
    @test size(buff.ψ[1]) == size(θ[1])
    @test size(buff.ψ[2]) == (num_actions,)
    d = p(buff, θ, s)
    @test eltype(d.μ) == T
    a = Array{T,1}([1.0, 0.0])
    logp1 = logpdf!(buff, p, θ, s, a)
    println(logpdf(d, a))
    @test typeof(logp1) == T
    @test logp1 == logpdf(p, θ, s, a)    

    
    gauto1 = gradient(θ->logpdf(p,θ,s,a),θ)[1]

    logp3 = grad_logpdf!(buff, p, θ, s, a)
    @test typeof(logp3) == T
    @test logp3 == logp1
    @test all(isapprox.(buff.ψ[1], gauto1[1]))
    @test all(isapprox.(buff.ψ[2], gauto1[2]))
    g2 = (Array{T,2}([0.0 0.0; 1.0 0.0; 2.0 0.0]), Array{T,1}([0.0, -1.0]))

    @test all(isapprox.(buff.ψ[1], g2[1]))
    @test all(isapprox.(buff.ψ[2], g2[2]))

    logp4 = sample_with_trace!(buff, p, θ, s)
    @test typeof(logp4) == T
    @test all(.!isapprox.(buff.action, 0.0))
    @test all(isapprox.(buff.ψ[1], s * buff.action'))
    @test all(isapprox.(buff.ψ[2], buff.action.^2 .- 1))
end
