using Policies
using Test


@testset "Policies.jl" begin
    @testset "SoftmaxTests" begin include("softmaxtests.jl") end
    @testset "NormalTests" begin include("normaltests.jl") end
end
