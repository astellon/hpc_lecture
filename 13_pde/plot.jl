using HDF5, Plots

function loadmat(path::AbstractString)
  raw = read(path, String)
  mat = parse.(Float64, hcat(split.(split(raw, "\n", keepempty=false), " ", keepempty=false)...))
end

u = h5read("u.h5", "dataset")
v = h5read("v.h5", "dataset")
p = h5read("p.h5", "dataset")

NX = size(u, 1)
NY = size(u, 2)

x = range(0, 2, length=NX)
y = range(0, 2, length=NY)

X = repeat(x, inner=NY)
Y = repeat(y, outer=NX)

plot()
contour!(x, y, p, fill=true)
quiver!(X, Y, quiver=(u[:].*0.1, v[:].*0.1))
savefig("result.png")
