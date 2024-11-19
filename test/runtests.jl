using LogHeightmaps
using Test
using Aqua
using JET

@testset "LogHeightmaps.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(LogHeightmaps)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(LogHeightmaps; target_defined_modules = true)
    end
    # Write your tests here.
end
