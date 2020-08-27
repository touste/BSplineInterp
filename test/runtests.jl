using BSplineInterp

using Images, OffsetArrays, BenchmarkTools

using Test


imdata(im::AbstractArray) = im
imdata(im::OffsetArray) = im.parent


imnames = ["small.png", "medium.png", "large.png"]

for imn in imnames

    imraw = Float32.(Gray.(load(joinpath(dirname(@__FILE__), imn))))

    for (name, im) in zip(["Regular Array", "Offset Array"], [imraw, OffsetArray(imraw, (10,10))])

        testname = name*" with image: "*imn
        @testset "$testname" begin

            itp = interpolate(im)

            itpval = itp(100.5, 100.5)
            @test isfinite(itpval)

            dx = 0.
            dy = 0.
            newim = [itp(x,y) for y in axes(im)[1] .+ dy, x in axes(im)[2] .+ dx]


            @test !(0 in (newim .≈ imdata(im))[begin+10:end-10,begin+10:end-10])

            gd = gradient(itp, 100.5, 100.5)
            @test isfinite(gd[1])
            @test isfinite(gd[2])

            itpgd = itpandgradient(itp, 100.5, 100.5)
            @test isfinite(itpgd[1])
            @test isfinite(itpgd[2])
            @test isfinite(itpgd[3])


            itp = interpolate!(itp, im)
            @btime interpolate!($itp, $im)
            @test itpval == itp(100.5, 100.5)

            @test itp(-10.,-10.) == 0

        end
    end

end


