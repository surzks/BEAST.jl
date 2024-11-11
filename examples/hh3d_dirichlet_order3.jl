using CompScienceMeshes, BEAST

o, x, y, z = euclidianbasis(3)


Γ = readmesh(joinpath(dirname(pathof(BEAST)),"../examples/sphere2.in"))

X = BEAST.lagrangecx(Γ; order=3)
# X = subdsurface(Γ)
# X = raviartthomas(Γ)
@show numfunctions(X)

κ = 1.0; γ = im*κ
a = Helmholtz3D.singlelayer(wavenumber=κ)
# b = Helmholtz3D.doublelayer_transposed(gamma=κ*im) +0.5Identity()

uⁱ = Helmholtz3D.planewave(wavenumber=κ, direction=z)
f = strace(uⁱ,Γ)
# g = ∂n(uⁱ)

BEAST.@defaultquadstrat (a,X,X) BEAST.DoubleNumSauterQstrat(7,8,6,6,6,6)
BEAST.@defaultquadstrat (f,X) BEAST.SingleNumQStrat(12)

@hilbertspace u
@hilbertspace v

# eq1 = @discretise a[v,u] == f[v] u∈X v∈X
# eq2 = @discretise b[v,u] == g[v] u∈X v∈X

A = assemble(a[v,u], X, X)
b = assemble(f[v], X)
x1 = AbstractMatrix(A) \ b
# x1 = BEAST.GMRESSolver(A; reltol=1e-10) * b

# x1 = gmres(eq1; tol=1e-6)
# x2 = gmres(eq2)

fcr1, geo1 = facecurrents(x1, X)
# fcr2, geo2 = facecurrents(x2, X)

# include(Pkg.dir("CompScienceMeshes","examples","plotlyjs_patches.jl"))
using LinearAlgebra
using Plotly
pt1 = Plotly.plot(patch(geo1, real.(norm.(fcr1))))
# pt1 = Plotly.plot(patch(geo1, ones(length(geo1))))
# pt2 = Plotly.plot(patch(geo2, real.(norm.(fcr2))));
display(pt1)

# using Test

# ## test the results
# Z = assemble(a,X,X);
# m1, m2 = 1, numfunctions(X)
# chm, chn = chart(Γ,cells(Γ)[m1]), chart(Γ,cells(Γ)[m2])
# ctm, ctn = CompScienceMeshes.center(chm), CompScienceMeshes.center(chn)
# R = norm(cartesian(ctm)-cartesian(ctn))
# G = exp(-im*κ*R)/(4π*R)
# Wmn = volume(chm) * volume(chn) * G
# @show abs(Wmn-Z[m1,m2]) / abs(Z[m1,m2])
# @test abs(Wmn-Z[m1,m2]) / abs(Z[m1,m2]) < 2.0e-3

# r = assemble(f,X)
# m1 = 1
# chm = chart(Γ,cells(Γ)[m1])
# ctm = CompScienceMeshes.center(chm)
# sm = volume(chm) * f(ctm)
# r[m1]
# @show abs(sm - r[m1]) / abs(r[m1])
# @test abs(sm - r[m1]) / abs(r[m1]) < 1e-3
