module BSplineInterp


export interpolate, interpolate!, gradient, itpandgradient





using FFTW
using LinearAlgebra
using StaticArrays
using OffsetArrays


const KERNEL = @SVector [1/120,13/60,11/20,13/60,1/120]

const QK = @SMatrix [1/120  13/60  11/20 13/60  1/120 0;
         -1/24   -5/12    0    5/12  1/24  0;
          1/12    1/6   -1/2   1/6   1/12  0;
         -1/12    1/6     0   -1/6   1/12  0;
          1/24   -1/6    1/4  -1/6   1/24  0;
         -1/120   1/24  -1/12  1/12 -1/24 1/120]




struct FFTbuf{T, Tplan}
    #Read only
    kernel_x_fft::Vector{Complex{T}}
    kernel_y_fft::Vector{Complex{T}}
    pcol::Tplan
    prow::Tplan
    #Buffers
    kernel_x_buf::Vector{Vector{Complex{T}}}
    kernel_y_buf::Vector{Vector{Complex{T}}}
    slice_buf::Vector{Vector{T}}
end


function FFTbuf(nl, nc, T)

    kernel_x = zeros(T, nc)
    kernel_x[1:3] .= KERNEL[3:5]
    kernel_x[end-1:end] .= KERNEL[1:2]
    pcol = plan_rfft(kernel_x, flags=FFTW.UNALIGNED)
    kernel_x_fft = Vector{Complex{T}}(undef, div(nc,2)+1)
    mul!(kernel_x_fft, pcol, kernel_x)

    kernel_y = zeros(T, nl)
    kernel_y[1:3] .= KERNEL[3:5]
    kernel_y[end-1:end] .= KERNEL[1:2]
    prow = plan_rfft(kernel_y, flags=FFTW.UNALIGNED)
    kernel_y_fft = Vector{Complex{T}}(undef, div(nl,2)+1)
    mul!(kernel_y_fft, prow, kernel_y)

    return FFTbuf{T, typeof(pcol)}(kernel_x_fft, kernel_y_fft, pcol, prow, [similar(kernel_x_fft) for _ in 1:Threads.nthreads()], [similar(kernel_y_fft) for _ in 1:Threads.nthreads()], [zeros(T, nc) for _ in 1:Threads.nthreads()])

end


struct BSInterp{Tdata, Tplan, Tbcoefs, Tcoefs, Tax}
    fftbuf::FFTbuf{Tdata, Tplan}
    bcoefs::Tbcoefs
    itpcoefs::Tcoefs
    axes::Tax
end











function interpolate(im::AbstractMatrix{T}) where T

    nl, nc = size(im)

    fftbuf = FFTbuf(nl, nc, T)
    bcoefs = similar(im)
    interpcoefs = similar(im, SMatrix{6,6,T,36})

    compute_bcoefs!(bcoefs, fftbuf, im)
    compute_interpcoefs!(interpcoefs, bcoefs)

    return BSInterp(fftbuf, bcoefs, interpcoefs, axes(im))
end


function interpolate(ofim::OffsetArray{T}) where T

    nl, nc = size(ofim)

    fftbuf = FFTbuf(nl, nc, T)
    bcoefs = similar(ofim)
    interpcoefs = similar(ofim, SMatrix{6,6,T,36})

    compute_bcoefs!(bcoefs.parent, fftbuf, ofim.parent)
    compute_interpcoefs!(interpcoefs.parent, bcoefs.parent)

    return BSInterp(fftbuf, bcoefs, interpcoefs, axes(ofim))
end



function interpolate!(itp::BSInterp, im::AbstractMatrix{T}) where T

    @assert size(im)==size(itp.bcoefs)

    compute_bcoefs!(itp.bcoefs, itp.fftbuf, im)
    compute_interpcoefs!(itp.interpcoefs, itp.bcoefs)

    return BSInterp(itp.fftbuf, itp.bcoefs, itp.interpcoefs, axes(im))
end


function interpolate!(itp::BSInterp, ofim::OffsetArray{T}) where T

    @assert size(ofim)==size(itp.bcoefs)

    compute_bcoefs!(itp.bcoefs.parent, itp.fftbuf, ofim.parent)
    compute_interpcoefs!(itp.interpcoefs.parent, itp.bcoefs.parent)

    return BSInterp(itp.fftbuf, itp.bcoefs, itp.interpcoefs, axes(ofim))
end





function compute_bcoefs!(bcoefs, fftbuf, im::AbstractMatrix{T}) where T

    nl, nc = size(im)

    Threads.@threads for i in 1:nl
        @views copy!(fftbuf.slice_buf[Threads.threadid()], im[i,:])
        mul!(fftbuf.kernel_x_buf[Threads.threadid()], fftbuf.pcol, fftbuf.slice_buf[Threads.threadid()])
        fftbuf.kernel_x_buf[Threads.threadid()] ./= fftbuf.kernel_x_fft
        ldiv!(fftbuf.slice_buf[Threads.threadid()], fftbuf.pcol, fftbuf.kernel_x_buf[Threads.threadid()])
        @views copy!(bcoefs[i,:], fftbuf.slice_buf[Threads.threadid()])
    end

    Threads.@threads for i in 1:nc
        @views mul!(fftbuf.kernel_y_buf[Threads.threadid()], fftbuf.prow, bcoefs[:,i])
        fftbuf.kernel_y_buf[Threads.threadid()] ./= fftbuf.kernel_y_fft
        @views ldiv!(bcoefs[:,i], fftbuf.prow, fftbuf.kernel_y_buf[Threads.threadid()])
    end

end


function compute_interpcoefs!(interpcoefs, bcoefs)
    nl, nc = size(bcoefs)
    Threads.@threads for j in 3:nc-3
        idx = -2:3
        @inbounds @simd for i in 3:nl-3
            @views interpcoefs[i,j] = QK*SMatrix{6,6}(bcoefs[i .+ idx, j .+ idx])*QK'
        end
    end
end


function (itp::BSInterp{Tdata, Tplan, Tbcoefs, Tcoefs, Tax})(x, y) where {Tdata, Tplan, Tbcoefs, Tcoefs, Tax}

    l, c = itp.axes
    
    xf = clamp(floor(Int, x), first(c), last(c))
    yf = clamp(floor(Int, y), first(l), last(l))
    dx = Tdata(x) - xf
    dy = Tdata(y) - yf
    dx2 = dx*dx
    dx3 = dx2*dx
    dx4 = dx3*dx
    dx5 = dx4*dx
    dy2 = dy*dy
    dy3 = dy2*dy
    dy4 = dy3*dy
    dy5 = dy4*dy
    xvec = SVector(1, dx, dx2, dx3, dx4, dx5)
    yvec = SVector(1, dy, dy2, dy3, dy4, dy5)

    return xvec'*itp.itpcoefs[yf, xf]*yvec

end



function gradient(itp::BSInterp{Tdata, Tplan, Tbcoefs, Tcoefs, Tax}, x, y) where {Tdata, Tplan, Tbcoefs, Tcoefs, Tax}

    l, c = itp.axes
    
    xf = clamp(floor(Int, x), first(c), last(c))
    yf = clamp(floor(Int, y), first(l), last(l))
    dx = Tdata(x) - xf
    dy = Tdata(y) - yf
    dx2 = dx*dx
    dx3 = dx2*dx
    dx4 = dx3*dx
    dx5 = dx4*dx
    dy2 = dy*dy
    dy3 = dy2*dy
    dy4 = dy3*dy
    dy5 = dy4*dy
    xvec = SVector(1, dx, dx2, dx3, dx4, dx5)
    yvec = SVector(1, dy, dy2, dy3, dy4, dy5)
    dxvec = SVector(0, 1, 2*dx, 3*dx2, 4*dx3, 5*dx4)
    dyvec = SVector(0, 1, 2*dy, 3*dy2, 4*dy3, 5*dy4)

    return dxvec'*itp.itpcoefs[yf, xf]*yvec, xvec'*itp.itpcoefs[yf, xf]*dyvec

end



function itpandgradient(itp::BSInterp{Tdata, Tplan, Tbcoefs, Tcoefs, Tax}, x, y) where {Tdata, Tplan, Tbcoefs, Tcoefs, Tax}

    l, c = itp.axes
    
    xf = clamp(floor(Int, x), first(c), last(c))
    yf = clamp(floor(Int, y), first(l), last(l))
    dx = Tdata(x) - xf
    dy = Tdata(y) - yf
    dx2 = dx*dx
    dx3 = dx2*dx
    dx4 = dx3*dx
    dx5 = dx4*dx
    dy2 = dy*dy
    dy3 = dy2*dy
    dy4 = dy3*dy
    dy5 = dy4*dy
    xvec = SVector(1, dx, dx2, dx3, dx4, dx5)
    yvec = SVector(1, dy, dy2, dy3, dy4, dy5)
    dxvec = SVector(0, 1, 2*dx, 3*dx2, 4*dx3, 5*dx4)
    dyvec = SVector(0, 1, 2*dy, 3*dy2, 4*dy3, 5*dy4)

    return xvec'*itp.itpcoefs[yf, xf]*yvec, dxvec'*itp.itpcoefs[yf, xf]*yvec, xvec'*itp.itpcoefs[yf, xf]*dyvec

end




end # module
