using BSplineInterp

using Images, OffsetArrays

using Test

const im = Float32.(Gray.(load(joinpath(dirname(@__FILE__), "speckle.png"))))

@testset "Regular Array" begin

    itp = interpolate(im)

    itpval = itp(100.5, 100.5)
    @test isfinite(itpval)

    dx = 0.
    dy = 0.
    newim = [itp(x,y) for x in axes(im)[2] .+ dx, y in axes(im)[1] .+ dy]


    @test !(0 in (newim .≈ im)[begin+10:end-10,begin+10:end-10])

    gd = gradient(itp, 100.5, 100.5)
    @test isfinite(gd[1])
    @test isfinite(gd[2])

    itpgd = itpandgradient(itp, 100.5, 100.5)
    @test isfinite(itpgd[1])
    @test isfinite(itpgd[2])
    @test isfinite(itpgd[3])


    itp = interpolate!(itp, im)
    @test itpval == itp(100.5, 100.5)

    @test itp(-10.,-10.) == 0

end



@testset "Offset Array" begin

    ofim = OffsetArray(im, (200,100))

    itp = interpolate(ofim)

    itpval = itp(200.5, 300.5)
    @test isfinite(itpval)

    dx = 0.
    dy = 0.
    newim = [itp(x,y) for x in axes(ofim)[2] .+ dx, y in axes(ofim)[1] .+ dy]

    @test !(0 in (newim .≈ ofim)[begin+10:end-10,begin+10:end-10])

    gd = gradient(itp, 200.5,300.5)
    @test isfinite(gd[1])
    @test isfinite(gd[2])

    itpgd = itpandgradient(itp, 200.5, 300.5)
    @test isfinite(itpgd[1])
    @test isfinite(itpgd[2])
    @test isfinite(itpgd[3])

    itp = interpolate!(itp, im)
    @test itpval == itp(200.5, 300.5)

    @test itp(-10.,-10.) == 0

end

