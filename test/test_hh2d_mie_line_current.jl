using BEAST
using CompScienceMeshes
using SpecialFunctions
using LinearAlgebra
using StaticArrays
using Test

const ε0 = 8.854187821e-12
const μ0 = 4π * 1e-7
const c0 = 1/sqrt(ε0*μ0)
const η0 = sqrt(μ0/ε0)

dbesselj(n,x) = (besselj(n-1, x)  - besselj(n+1, x)) / 2
dhankelh2(n,x) = (hankelh2(n-1, x)  - hankelh2(n+1, x)) / 2

cart2polar(x,y) = SVector(sqrt(x^2 + y^2), atan(y, x))

##
f = 1.0e9 # frequency
λ = c0/f # wavelength
k = 2π/λ # wavenumber
ω = 2π*f # angular frequency
h = λ/10 # mesh segment size
a = 1.0 # Radius of the cylindrical scatterer
rc = 500.0*a # points on which field is to be calculated

I = 1.0 # 1 A
ρp = 2.0 # Source radial position
φp = 0.0 # Source angular position
xp = ρp * cos(φp)
yp = ρp * sin(φp)


circle = CompScienceMeshes.meshcircle(a,h)
X0 = lagrangecxd0(circle)
X1 = lagrangec0d1(circle)
pts = meshcircle(rc, 0.6*rc).vertices

##
# TM Scattering

#=
 Analytical solutions

 This follows the Section 6.5.3, Jin, Theory and Computation of EM fields. Particulary, Equation 6.5.38 is
 used to obtain the scattered fields.
=#

function TM_pec_line_curr(I, k, a, ρ, φ, ρp, φp)

    A_d(n) = besselj(n,k*a)/hankelh2(n,k*a)

    valEz(n) = A_d(n) * hankelh2(n,k*ρ) * hankelh2(n,k*ρp) * exp(im*n*(φ-φp))
    valHρ(n) = -1/(im * ω * μ0)  * (im * n / ρ) * A_d(n) * hankelh2(n,k*ρp) * hankelh2(n,k*ρ) * exp(im*n*(φ-φp))
    valHφ(n) = 1/(im * ω * μ0) * k * A_d(n) * hankelh2(n,k*ρp) * dhankelh2(n,k*ρ) * exp(im*n*(φ-φp))

    retEz = valEz(0)
    retHρ = valHρ(0)
    retHφ = valHφ(0)

    n = 0
    while true
        n += 1
        bufEz = valEz(n) + valEz(-n)
        if abs(bufEz) >= abs(retEz) * eps(eltype(real(retEz))) * 1e3
            retEz += bufEz
        else
            break
        end
    end

    n = 0
    while true
        n += 1
        bufHρ = valHρ(n) + valHρ(-n)
        if abs(bufHρ) >= abs(retHρ) * eps(eltype(real(retHρ))) * 1e3
            retHρ += bufHρ
        else
            break
        end
    end

    n = 0
    while true
        n += 1
        bufHφ = valHφ(n) + valHφ(-n)
        if abs(bufHφ) >= abs(retHφ) * eps(eltype(real(retHφ))) * 1e3
            retHφ += bufHφ
        else
            break
        end
    end

    Hₜ = retHρ * SVector(cos(φ),sin(φ)) + retHφ * SVector(-sin(φ),cos(φ))

    return η0*k*I/4 * retEz, η0*k*I/4 * Hₜ
end

function TM_pec_line_curr_E(I, k, a, pts, spt)
    return [TM_pec_line_curr(I,k,a,cart2polar(p[1],p[2])..., spt[1],spt[2])[1] for p in pts]
end

function TM_pec_line_curr_H(I, k, a, pts, spt)
    return [TM_pec_line_curr(I,k,a,cart2polar(p[1],p[2])..., spt[1],spt[2])[2] for p in pts]
end


# Numerical
# LHS
𝒮 = Helmholtz2D.singlelayer(; alpha=im*k*η0, wavenumber=k)
M_TMEFIE = assemble(𝒮, X0, X0)

# TM-MFIE system matrix
𝒟ᵀ = Helmholtz2D.doublelayer_transposed(; wavenumber=k)
Dᵀ = assemble(𝒟ᵀ, X0, X0)

I0 = assemble(BEAST.Identity(), X0, X0)
M_TMMFIE = +0.5*I0 + Dᵀ

# RHS
# Choosing the amplitude of Einc such that [Eq 6.5.11, Jin] is satisfied
Ez_Einc = Helmholtz2D.monopole(;position = SVector(xp,yp), wavenumber=k, amplitude = -η0*k*I*im)
ez_Einc = assemble(DirichletTrace(Ez_Einc),X0)
j_TMEFIE = M_TMEFIE \ ez_Einc

Ez_sca_num = -potential(HH2DSingleLayerNear(𝒮),pts , j_TMEFIE, X0;type=ComplexF64)

Ez_sca_ana = TM_pec_line_curr_E(I, k, a, pts, SVector(ρp,φp))

@test norm(Ez_sca_num - Ez_sca_ana) / norm(Ez_sca_ana) < 0.003


Ht_sca_num = -potential(HH2DDoubleLayerTransposedNear(𝒟ᵀ), pts, j_TMEFIE, X0; type=SVector{2,ComplexF64})

Ht_sca_ana = TM_pec_line_curr_H(I, k, a, pts, SVector(ρp,φp))

@test norm(Ref(ẑ) .× Ht_sca_num - Ht_sca_ana) / norm(Ht_sca_ana) < 0.003

Ht_sca_num = -potential(HH2DDoubleLayerTransposedNear(𝒟ᵀ), pts, j_TMEFIE, X0; type=SVector{2, ComplexF64})
norm(Ref(ẑ) .× Ht_sca_num - Ht_sca_ana) ./ norm(Ht_sca_ana)

# usage of the TMMFIE current results in a loss of accuracy
Ht_inc = -1.0 / (im * ω * μ0) * curl(Ez_Einc)
𝗵t = assemble(TangentTrace(Ht_inc), X0)
𝗷_TMMFIE = M_TMMFIE \ 𝗵t

Ht_sca_num = -potential(HH2DDoubleLayerTransposedNear(𝒟ᵀ), pts, 𝗷_TMMFIE, X0; type=SVector{2, ComplexF64})
norm(Ref(ẑ) .× Ht_sca_num - Ht_sca_ana) ./ norm(Ht_sca_ana)

# Observation :error increases as the distance between r' and r reduces, i.e., when we make rc smaller.
##
# TE scattering

#=
 Here, the current source is still infinitely long along the z-direction, but the current is directed in
 the ϕ-direction. The expressions for the scattered field are derived stating that the Hz field satisfies
 the Helmholtz equation and thus can be expanded in terms of harmonics.


 Let's first compare the incident fields. To do so, we place the source at ρ = 2.0 and φ = 0.0, and a
 circle of radius r_inc = 1.0 with it's centre at the origin. We compute the incident magnetic fields on the
 circumference of the circle, both analytically and numerically and compare the two, which, naturally, are
 supposed to match.
=#
I = 1.0
ρp = 2.0
φp = 0.0
r_inc = 1.0

pts_inc =  meshcircle(r_inc, 0.6*r_inc).vertices

function Hz_inc(I,k,ρ,φ,ρp,φp)

    val(n) = dhankelh2(n,k*ρp) * besselj(n,k*ρ) * exp(im*n*φ)
    ret = val(0)

    n = 0
    while true
        n += 1
        buf = val(n) + val(-n)
        if abs(buf) >= abs(ret) * eps(eltype(real(ret))) * 1e3
            ret += buf
        else
            break
        end
    end
    prefac = k^2/(μ0)
    return  prefac * -I*k/(4*im) * ret
end

function Hz_inc(I::F, k::F, pts::Vector{SVector{2,F}}, spt::SVector{2,F}) where F
    return [Hz_inc(I, k, cart2polar(p[1],p[2])..., spt[1], spt[2]) for p in pts]
end

Hz_inc_ana = Hz_inc(I, k, pts_inc, SVector(ρp, φp))

# Numerical
xp = ρp * cos(φp)
yp = ρp * sin(φp)

# amplitude = k^2/mu0 to make it a magnetic monopole
Hz_inc_num = Helmholtz2D.directedmonopole(;position = SVector(xp, yp),
    direction = SVector(0.0,1.0),
    wavenumber = k,
    amplitude = k^2/μ0)

@test norm( Hz_inc_ana - Hz_inc_num.(pts_inc)) /norm(Hz_inc_ana) ≈ 0 atol = 1e-12

##
# Scattered field
# TODO: Add ρ-directed current's contribution

function An_TE(a::F, n::I; Z = ComplexF64(0)) where {F,I}
    x = k*a
    # TODO: extend to include the impedance BC part
    An_TE = dhankelh2(n,x)/dbesselj(n,x)
    return -An_TE
end

function TE_pec_line_curr(I::F,k::F,a::F,ρ::F,φ::F,ρp::F,φp::F; Z = ComplexF64(0)) where F
    
    valHz(n) = dhankelh2(n,k*ρp)/An_TE(a,n) * hankelh2(n,k*ρ) * exp(im*n*(φ-φp))
    valEρ(n) = -1 / (im * ω * ε0)  * (im * n / ρ) * dhankelh2(n,k*ρp)/An_TE(a,n) * hankelh2(n,k*ρ) * exp(im*n*(φ-φp))
    valEφ(n) =  1 / (im * ω * ε0)  * k *  dhankelh2(n,k*ρp)/An_TE(a,n) * dhankelh2(n,k*ρ) * exp(im*n*(φ-φp))
    

    retHz = valHz(0)
    retEρ = valEρ(0)
    retEφ = valEφ(0)

    n = 0
    while true
        n += 1
        bufHz = valHz(n) + valHz(-n)

        if abs(bufHz) >= abs(retHz) * eps(eltype(real(retHz))) * 1e3
            retHz += bufHz
        else
            break
        end
    end

    n = 0
    while true
        n += 1
        bufEρ = valEρ(n) + valEρ(-n)

        if abs(bufEρ) >= abs(retEρ) * eps(eltype(real(retEρ))) * 1e3
            retEρ += bufEρ
        else
            break
        end
    end

    n = 0
    while true
        n += 1
        bufEφ = valEφ(n) + valEφ(-n)

        if abs(bufEφ) >= abs(retEφ) * eps(eltype(real(retEφ))) * 1e3
            retEφ += bufEφ
        else
            break
        end
    end

    prefac = k^2/(μ0)
    return  prefac * -I*k/(4*im) * retHz, I*k^3/(4*im*μ0) * (retEρ * SVector(cos(φ),sin(φ)) + retEφ * SVector(-sin(φ),cos(φ)))
end

function TE_pec_line_curr_H(I, k, a, pts, spt)
    return [TE_pec_line_curr(I,k,a,cart2polar(p[1],p[2])..., spt[1],spt[2])[1] for p in pts]
end

function TE_pec_line_curr_E(I, k, a, pts, spt)
    return [TE_pec_line_curr(I,k,a,cart2polar(p[1],p[2])..., spt[1],spt[2])[2] for p in pts]
end

##
# Numerical

# TE_MFIE
# LHS
𝒟 = Helmholtz2D.doublelayer(;wavenumber = k)
D = assemble(𝒟, X1, X1)
I1 = assemble(BEAST.Identity(), X1, X1)
TEMFIE = 0.5I1 - D

# RHS
hz_inc = -assemble(DirichletTrace(Hz_inc_num),X1)
j_TEMFIE = TEMFIE \ hz_inc

Hz_sca_ana = TE_pec_line_curr_H(I, k, a, pts, SVector(ρp, φp))

Hz_sca_num = -potential(HH2DDoubleLayerNear(𝒟), pts, j_TEMFIE, X1; type=ComplexF64)
@test norm( Hz_sca_ana - Hz_sca_num) /norm(Hz_sca_ana) <= 0.003

##
# TE-EFIE
# LHS
𝒩 = Helmholtz2D.hypersingular(;alpha=im*ω*μ0, beta=1.0/(im*ω*ε0), wavenumber=k)
TEEFIE = assemble(𝒩, X1, X1)

# RHS with j_TEMFIE
Et_sca_num = potential(HH2DHyperSingularNear(𝒩), pts, j_TEMFIE, X1; type=SVector{2, ComplexF64})
Et_sca_num_curl = Ref(ẑ) .× Et_sca_num

Et_sca_ana = TE_pec_line_curr_E(I, k, a, pts, SVector(ρp, φp))

@test norm( Et_sca_num_curl - Et_sca_ana) ./ norm(Et_sca_ana) <= 0.003

# RHS with j_TEEFIE
Et_inc_num = 1.0/(im*ω*ε0) * curl(Hz_inc_num)
et_inc = -assemble(TangentTrace(Et_inc_num),X1)
j_TEEFIE = TEEFIE \  et_inc

@test norm(j_TEEFIE - j_TEMFIE) ./ norm(j_TEMFIE) <= 0.01

Et_sca_num_2 = potential(HH2DHyperSingularNear(𝒩), pts, j_TEEFIE, X1; type=SVector{2, ComplexF64})
Et_sca_num_2_curl = Ref(ẑ) .× Et_sca_num_2
@test norm(Et_sca_num_2_curl - Et_sca_ana) ./ norm(Et_sca_ana) <= 0.003
##