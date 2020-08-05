using SafeTestsets



@safetestset "SoftmaxTests" begin include("softmaxtests.jl") end
@safetestset "NormalTests" begin include("normaltests.jl") end
