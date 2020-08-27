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




struct FFTbuf{T, Tplanrow, Tplancol}
    # Read only
    kernel_row_fft::Vector{Complex{T}} # FFT of row kernel
    kernel_col_fft::Vector{Complex{T}} # FFT of column kernel
    prow::Tplanrow # FFT plan for image rows
    pcol::Tplancol # FFT plan for image columns
    # Buffers
    buf_kernel_row_fft::Vector{Vector{Complex{T}}} # Pre-allocated FFT row buffer
    buf_kernel_col_fft::Vector{Vector{Complex{T}}} # Pre-allocated FFT column buffer
    slice_row_buf::Vector{Vector{T}} # Pre-allocated row buffer
end


function FFTbuf(im::AbstractMatrix{T}) where T

    nl, nc = size(im)

    kernel_row = zeros(T, nc)
    kernel_row[1:3] .= KERNEL[3:5]
    kernel_row[end-1:end] .= KERNEL[1:2]
    kernel_row_fft = rfft(kernel_row)

    kernel_col = zeros(T, nl)
    kernel_col[1:3] .= KERNEL[3:5]
    kernel_col[end-1:end] .= KERNEL[1:2]
    kernel_col_fft = rfft(kernel_col)

    FFTW.set_num_threads(1)
    pcol = plan_rfft(kernel_col, flags=FFTW.UNALIGNED)
    prow = plan_rfft(kernel_row, flags=FFTW.UNALIGNED)

    return FFTbuf{T, typeof(prow), typeof(pcol)}(kernel_row_fft, kernel_col_fft, prow, pcol, [similar(kernel_row_fft) for _ in 1:Threads.nthreads()], [similar(kernel_col_fft) for _ in 1:Threads.nthreads()], [similar(kernel_row) for _ in 1:Threads.nthreads()])

end


struct BSInterp{Tdata, Tplanrow, Tplancol, Tbcoefs, Tcoefs, Tax}
    fftbuf::FFTbuf{Tdata, Tplanrow, Tplancol}
    bcoefs::Tbcoefs
    itpcoefs::Tcoefs
    axes::Tax
end











function interpolate(im::AbstractMatrix{T}) where T

    fftbuf = FFTbuf(im)
    bcoefs = similar(im)
    itpcoefs = similar(im, SMatrix{6,6,T,36})

    compute_bcoefs!(bcoefs, fftbuf, im)
    compute_itpcoefs!(itpcoefs, bcoefs)

    return BSInterp(fftbuf, bcoefs, itpcoefs, axes(im))
end


function interpolate(ofim::OffsetArray{T}) where T

    fftbuf = FFTbuf(ofim)
    bcoefs = similar(ofim)
    itpcoefs = similar(ofim, SMatrix{6,6,T,36})

    compute_bcoefs!(bcoefs.parent, fftbuf, ofim.parent)
    compute_itpcoefs!(itpcoefs.parent, bcoefs.parent)

    return BSInterp(fftbuf, bcoefs, itpcoefs, axes(ofim))
end



function interpolate!(itp::BSInterp, im::AbstractMatrix{T}) where T

    @assert size(im)==size(itp.bcoefs)

    compute_bcoefs!(itp.bcoefs, itp.fftbuf, im)
    compute_itpcoefs!(itp.itpcoefs, itp.bcoefs)

    return BSInterp(itp.fftbuf, itp.bcoefs, itp.itpcoefs, axes(im))
end


function interpolate!(itp::BSInterp, ofim::OffsetArray{T}) where T

    @assert size(ofim)==size(itp.bcoefs)

    compute_bcoefs!(itp.bcoefs.parent, itp.fftbuf, ofim.parent)
    compute_itpcoefs!(itp.itpcoefs.parent, itp.bcoefs.parent)

    return BSInterp(itp.fftbuf, itp.bcoefs, itp.itpcoefs, axes(ofim))
end





function compute_bcoefs!(bcoefs, fftbuf, im::AbstractMatrix{T}) where T

    nl, nc = size(im)

    # Deconvolution along rows
    # As we work along rows, it is more efficient to place each row in a vector first,
    # process it then place it back to have contiguous memory
    Threads.@threads for i in 1:nl
        @views copy!(fftbuf.slice_row_buf[Threads.threadid()], im[i,:])
        mul!(fftbuf.buf_kernel_row_fft[Threads.threadid()], fftbuf.prow, fftbuf.slice_row_buf[Threads.threadid()])
        fftbuf.buf_kernel_row_fft[Threads.threadid()] ./= fftbuf.kernel_row_fft
        ldiv!(fftbuf.slice_row_buf[Threads.threadid()], fftbuf.prow, fftbuf.buf_kernel_row_fft[Threads.threadid()])
        @views copy!(bcoefs[i,:], fftbuf.slice_row_buf[Threads.threadid()])
    end

    # Deconvolution along columns
    Threads.@threads for i in 1:nc
        @views mul!(fftbuf.buf_kernel_col_fft[Threads.threadid()], fftbuf.pcol, bcoefs[:,i])
        fftbuf.buf_kernel_col_fft[Threads.threadid()] ./= fftbuf.kernel_col_fft
        @views ldiv!(bcoefs[:,i], fftbuf.pcol, fftbuf.buf_kernel_col_fft[Threads.threadid()])
    end

end


function compute_itpcoefs!(itpcoefs, bcoefs)
    nl, nc = size(bcoefs)
    Threads.@threads for j in 3:nc-3
        idx = -2:3
        @inbounds @simd for i in 3:nl-3
            @views itpcoefs[i,j] = QK*SMatrix{6,6}(bcoefs[i .+ idx, j .+ idx])*QK'
        end
    end
end


function (itp::BSInterp{Tdata,  Tplanrow, Tplancol, Tbcoefs, Tcoefs, Tax})(x, y) where {Tdata,  Tplanrow, Tplancol, Tbcoefs, Tcoefs, Tax}

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



function gradient(itp::BSInterp{Tdata,  Tplanrow, Tplancol, Tbcoefs, Tcoefs, Tax}, x, y) where {Tdata,  Tplanrow, Tplancol, Tbcoefs, Tcoefs, Tax}

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



function itpandgradient(itp::BSInterp{Tdata,  Tplanrow, Tplancol, Tbcoefs, Tcoefs, Tax}, x, y) where {Tdata,  Tplanrow, Tplancol, Tbcoefs, Tcoefs, Tax}

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
