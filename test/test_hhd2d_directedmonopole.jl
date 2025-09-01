using BEAST
using CompScienceMeshes
using StaticArrays
using LinearAlgebra
using Test
using SpecialFunctions
    
    
    ε0 = 8.854187821e-12
    μ0 = 4π*1e-7
    c0 = 1/sqrt(ε0*μ0)
    η0 = sqrt(μ0/ε0)

    f = 1e9 # 1 GHz
    ω = 2π*f
    λ = c0/f
    k = 2π/λ
    h = λ/10
    a = 1.0 # radius of the scatterer

    circle = CompScienceMeshes.meshcircle(a, h)
    X0 = lagrangecxd0(circle)
    X1 = lagrangec0d1(circle)

    # Computing the fields on a circle of radius rc
    rc = 500.0*a
    pts = meshcircle(rc, 0.3 * rc).vertices

    ## Analytical solutions

    dbesselj(n,x) = (besselj(n-1, x)  - besselj(n+1, x)) / 2
    dhankelh2(n,x) = (hankelh2(n-1, x)  - hankelh2(n+1, x)) / 2

    cart2polar(x,y) = SVector(sqrt(x^2 + y^2), atan(y, x))

    𝒟 = Helmholtz2D.doublelayer(; wavenumber=k)
    D = assemble(𝒟, X1, X1)
    I1 = assemble(BEAST.Identity(), X1, X1)
    M_TEMFIE = 0.5I1 - D

    # TE-EFIE
    𝒩 = Helmholtz2D.hypersingular(;alpha=im*ω*μ0, beta=1.0/(im*ω*ε0), wavenumber=k)
    M_TEEFIE = assemble(𝒩, X1, X1)

    ##
    # 1. Excitation: Planewave
    H0 = 1.0
    Hz_dm_inc = Helmholtz2D.directedmonopole(; 
        amplitude=H0,
        wavenumber=k,
        position=SVector(2.0, 0.0),
        direction=SVector(1.0, 0.0)
    )

    hz_dm_inc = -assemble(DirichletTrace(Hz_dm_inc), X1)
    j_TEMFIE_dm = M_TEMFIE \ hz_dm_inc

    Et_dm_inc = 1 / (im * ω * ε0) * curl(Hz_dm_inc)
    et_dm_inc = assemble(TangentTrace(Et_dm_inc), X1)

    j_TEEFIE_dm = M_TEEFIE \ et_dm_inc

    @test norm(j_TEEFIE_dm + j_TEMFIE_dm)/norm(j_TEMFIE_dm) < 0.01