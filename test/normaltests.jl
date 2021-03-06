using DecisionMakingPolicies
using Test

import Zygote: gradient
import Distributions: Normal, MvNormal

@testset "Stateless Normal Tests" begin
    num_actions = 1
    T = Float32
    p = StatelessNormal(T, num_actions)
    @test eltype(p.μ) == T
    @test eltype(p.σ) == T
    @test size(p.μ) == (num_actions,)
    @test size(p.σ) == (num_actions,)
    d = p()
    a = T(1.0)
    logp1 = logpdf(d, a)
    @test typeof(d) <: Normal{T}
    @test typeof(logp1) == T

    num_actions = 2
    p = StatelessNormal(T, num_actions)
    @test typeof(p()) <: MvNormal

    a = rand(p())
    @test eltype(a) == T
    @test size(a) == (2,)
    a = Array{T,1}([1.0, 0.0])
    d = p()
    logp1 = logpdf(d, a)
    logp2 = logpdf(p, a)
    @test typeof(logp2) == T
    @test_broken typeof(logp2) == typeof(logp1)
    @test isapprox(logp1, logp2)
    g = (μ=zero(p.μ), σ=zero(p.σ))
    gauto1 = gradient(p->logpdf(p(),a),p)[1]
    gauto2 = gradient(p->logpdf(p,a),p)[1]

    @test_broken eltype(gauto1.μ) == eltype(gauto2.σ)
    @test_broken eltype(gauto1.σ) == eltype(gauto2.σ)
    @test eltype(gauto2.μ) == eltype(p.μ)
    @test eltype(gauto2.σ) == eltype(p.σ)
    @test all(isapprox.(gauto1.μ, gauto2.μ))
    @test all(isapprox.(gauto1.σ, gauto2.σ))

    logp3 = grad_logpdf!(g, p, a)
    @test typeof(logp3) == T
    @test logp3 == logp2
    @test all(isapprox.(g[1], gauto1[1]))
    @test all(isapprox.(g[2], gauto1[2]))
    g2 = (μ=Array{T,1}([1.0, 0.0]), σ=Array{T,1}([0.0, -1.0]))

    @test all(isapprox.(g.μ, g2.μ))
    @test all(isapprox.(g.σ, g2.σ))

    A = zeros(T,2)
    A2 = zeros(T,(2,2))
    logp4 = sample_with_trace!(g, A, p)
    @test all(.!isapprox.(A, 0.0))
    @test all(isapprox.(g[1], A))
    @test all(isapprox.(g[2], A.^2 .- 1))

    sample_with_trace!(g, view(A2,:,1), p)
    @test all(.!isapprox.(A2[:,1], 0.0))
    @test all(isapprox.(A2[:,2], 0.0))
    a1 = A2[:,1]
    sample_with_trace!(g, view(A2,:, 2), p)
    @test all(isapprox.(A2[:, 1], a1))
    @test all(.!isapprox.(A2[:,2], 0.0))
end

@testset "Stateless Normal Buffer Tests" begin
    num_actions = 2
    T = Float32
    p = StatelessNormal(T, num_actions)
    buff = NormalBuffer(p)
    @test eltype(buff.μ) == T
    @test eltype(buff.action) == T
    @test eltype(buff.ψ.μ) == T
    @test eltype(buff.ψ.σ) == T
    @test size(buff.μ) == (num_actions,)
    @test size(buff.action) == (num_actions,)
    @test size(buff.ψ.σ) == (num_actions,)
    @test size(buff.ψ.μ) == (num_actions,)
    d = p(buff)
    @test eltype(d.μ) == T
    a = Array{T,1}([1.0, 0.0])
    logp1 = logpdf!(buff, p, a)
    @test typeof(logp1) == T

    
    
    g = (μ=similar(p.μ), σ=zero(p.σ))
    gauto1 = gradient(p->logpdf(p,a),p)[1]
    logp3 = grad_logpdf!(buff, p, a)
    @test typeof(logp3) == T
    @test logp3 == logp1
    @test all(isapprox.(buff.ψ.μ, gauto1.μ))
    @test all(isapprox.(buff.ψ.σ, gauto1.σ))
    g2 = (Array{T,1}([1.0, 0.0]), Array{T,1}([0.0, -1.0]))

    @test all(isapprox.(buff.ψ.μ, g2[1]))
    @test all(isapprox.(buff.ψ.σ, g2[2]))

    logp4 = sample_with_trace!(buff, p)
    @test typeof(logp4) == T
    @test all(.!isapprox.(buff.action, 0.0))
    @test all(isapprox.(buff.ψ.μ, buff.action))
    @test all(isapprox.(buff.ψ.σ, buff.action.^2 .- 1))
end


@testset "Linear Normal Tests" begin
    num_actions = 1
    num_features = 3
    T = Float32
    p = LinearNormal(T, num_features, num_actions)
    s = Array{T, 1}([0.0, 1.0, 2.0])

    @test eltype(p.W) == T
    @test eltype(p.σ) == T
    @test size(p.W) == (num_features, num_actions)
    @test size(p.σ) == (num_actions,)
    @test typeof(p(s)) <: Normal

    num_actions = 2
    p = LinearNormal(T, num_features, num_actions)
    @test typeof(p(s)) <: MvNormal

    a = rand(p(s))
    @test eltype(a) == T
    @test size(a) == (2,)
    a = Array{T,1}([1.0, 0.0])

    logp1 = logpdf(p(s), a)
    logp2 = logpdf(p, s, a)

    @test typeof(logp2) == T
    @test_broken typeof(logp2) == typeof(logp1)
    @test isapprox(logp1, logp2)
    g = (W=similar(p.W), σ=similar(p.σ))
    gauto1 = gradient(p->logpdf(p(s),a),p)[1]
    gauto2 = gradient(p->logpdf(p,s,a),p)[1]

    @test_broken eltype(gauto1.W) == eltype(gauto2.W)
    @test_broken eltype(gauto1.σ) == eltype(gauto2.σ)
    @test eltype(gauto2.W) == eltype(p.W)
    @test eltype(gauto2.σ) == eltype(p.σ)
    @test all(isapprox.(gauto1.W, gauto2.W))
    @test all(isapprox.(gauto1.σ, gauto2.σ))

    logp3 = grad_logpdf!(g, p, s, a)
    @test typeof(logp3) == T
    @test logp3 == logp2
    @test all(isapprox.(g.W, gauto1.W))
    @test all(isapprox.(g.σ, gauto1.σ))
    g2 = (Array{T,2}([0.0 0.0; 1.0 0.0; 2.0 0.0]), Array{T,1}([0.0, -1.0]))

    @test all(isapprox.(g.W, g2[1]))
    @test all(isapprox.(g.σ, g2[2]))

    A = zeros(T,2)
    A2 = zeros(T,(2,2))
    logp4 = sample_with_trace!(g, A, p, s)
    @test all(.!isapprox.(A, 0.0))
    @test all(isapprox.(g.W, s * A'))
    @test all(isapprox.(g.σ, A.^2 .- 1))

    sample_with_trace!(g, view(A2,:,1), p, s)
    @test all(.!isapprox.(A2[:,1], 0.0))
    @test all(isapprox.(A2[:,2], 0.0))
    a1 = A2[:,1]
    sample_with_trace!(g, view(A2,:, 2), p, s)
    @test all(isapprox.(A2[:, 1], a1))
    @test all(.!isapprox.(A2[:,2], 0.0))
end


@testset "Linear Normal Buffer Tests" begin
    num_actions = 2
    num_features = 3
    T = Float32
    p = LinearNormal(T, num_features, num_actions)
    s = Array{T, 1}([0.0, 1.0, 2.0])
    buff = NormalBuffer(p)
    @test eltype(buff.μ) == T
    @test eltype(buff.action) == T
    @test eltype(buff.ψ.W) == T
    @test eltype(buff.ψ.σ) == T
    @test size(buff.μ) == (num_actions,)
    @test size(buff.action) == (num_actions,)
    @test size(buff.ψ.W) == size(p.W)
    @test size(buff.ψ.σ) == (num_actions,)
    d = p(buff, s)
    @test eltype(d.μ) == T
    a = Array{T,1}([1.0, 0.0])
    logp1 = logpdf!(buff, p, s, a)
    println(logpdf(d, a))
    @test typeof(logp1) == T
    @test logp1 == logpdf(p, s, a)    

    
    gauto1 = gradient(p->logpdf(p,s,a),p)[1]

    logp3 = grad_logpdf!(buff, p, s, a)
    @test typeof(logp3) == T
    @test logp3 == logp1
    @test all(isapprox.(buff.ψ.W, gauto1.W))
    @test all(isapprox.(buff.ψ.σ, gauto1.σ))
    g2 = (Array{T,2}([0.0 0.0; 1.0 0.0; 2.0 0.0]), Array{T,1}([0.0, -1.0]))

    @test all(isapprox.(buff.ψ.W, g2[1]))
    @test all(isapprox.(buff.ψ.σ, g2[2]))

    logp4 = sample_with_trace!(buff, p, s)
    @test typeof(logp4) == T
    @test all(.!isapprox.(buff.action, 0.0))
    @test all(isapprox.(buff.ψ[1], s * buff.action'))
    @test all(isapprox.(buff.ψ[2], buff.action.^2 .- 1))
end
