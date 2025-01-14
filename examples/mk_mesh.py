#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import gmsh

import numpy as np

gmsh.initialize()

gmsh.model.add('topo')

mesh_size = 2.0
mesh_size_out = 20.0
out_lim = 100.0

# Surface

p01 = gmsh.model.geo.addPoint(-out_lim, -out_lim, 0, mesh_size_out)
p02 = gmsh.model.geo.addPoint(out_lim, -out_lim, 0, mesh_size_out)
p03 = gmsh.model.geo.addPoint(out_lim, out_lim, 0, mesh_size_out)
p04 = gmsh.model.geo.addPoint(-out_lim, out_lim, 0, mesh_size_out)

p01d = gmsh.model.geo.addPoint(-out_lim, -out_lim, -out_lim, mesh_size_out)
p02d = gmsh.model.geo.addPoint(out_lim, -out_lim, -out_lim, mesh_size_out)
p03d = gmsh.model.geo.addPoint(out_lim, out_lim, -out_lim, mesh_size_out)
p04d = gmsh.model.geo.addPoint(-out_lim, out_lim, -out_lim, mesh_size_out)

l01 = gmsh.model.geo.addLine(p01, p02)
l02 = gmsh.model.geo.addLine(p02, p03)
l03 = gmsh.model.geo.addLine(p03, p04)
l04 = gmsh.model.geo.addLine(p04, p01)

cl01 = gmsh.model.geo.addCurveLoop((l01, l02, l03, l04))

p05 = gmsh.model.geo.addPoint(-50.0, -50.0, 0, mesh_size)
p06 = gmsh.model.geo.addPoint(50.0, -50.0, 0, mesh_size)
p07 = gmsh.model.geo.addPoint(50.0, 50.0, 0, mesh_size)
p08 = gmsh.model.geo.addPoint(-50.0, 50.0, 0, mesh_size)

p05d = gmsh.model.geo.addPoint(-50.0, -50.0, -50, 2*mesh_size)
p06d = gmsh.model.geo.addPoint(50.0, -50.0, -50, 2*mesh_size)
p07d = gmsh.model.geo.addPoint(50.0, 50.0, -50, 2*mesh_size)
p08d = gmsh.model.geo.addPoint(-50.0, 50.0, -50, 2*mesh_size)

p09 = gmsh.model.geo.addPoint(-50.0, 0.0, 0, mesh_size)
p10 = gmsh.model.geo.addPoint(-40.0, 0.0, 0, mesh_size)
p11 = gmsh.model.geo.addPoint(-10.0, 0.0, 3, mesh_size)
p12 = gmsh.model.geo.addPoint(30.0, 0.0, 0, mesh_size)
p13 = gmsh.model.geo.addPoint(50.0, 0.0, 0, mesh_size)

l05 = gmsh.model.geo.addLine(p09, p05)
l06 = gmsh.model.geo.addLine(p05, p06)
l07 = gmsh.model.geo.addLine(p06, p13)
l08 = gmsh.model.geo.addSpline((p13, p12, p11, p10, p09))

cl02 = gmsh.model.geo.addCurveLoop((l05, l06, l07, l08))

l09 = gmsh.model.geo.addLine(p09, p08)
l10 = gmsh.model.geo.addLine(p08, p07)
l11 = gmsh.model.geo.addLine(p07, p13)

cl03 = gmsh.model.geo.addCurveLoop((l09, l10, l11, l08))

s01 = gmsh.model.geo.addPlaneSurface((cl01, cl02, cl03))
s02 = gmsh.model.geo.addSurfaceFilling((cl02,))
s03 = gmsh.model.geo.addSurfaceFilling((cl03,))

l12 = gmsh.model.geo.addLine(p05, p05d)
l13 = gmsh.model.geo.addLine(p06, p06d)
l14 = gmsh.model.geo.addLine(p07, p07d)
l15 = gmsh.model.geo.addLine(p08, p08d)

l16 = gmsh.model.geo.addLine(p05d, p06d)
l17 = gmsh.model.geo.addLine(p06d, p07d)
l18 = gmsh.model.geo.addLine(p07d, p08d)
l19 = gmsh.model.geo.addLine(p08d, p05d)

cl04 = gmsh.model.geo.addCurveLoop((l12, l16, -l13, -l06))
cl05 = gmsh.model.geo.addCurveLoop((l13, l17, -l14, l11, -l07))
cl06 = gmsh.model.geo.addCurveLoop((l14, l18, -l15, l10))
cl07 = gmsh.model.geo.addCurveLoop((l15, l19, -l12, -l05, l09))
cl08 = gmsh.model.geo.addCurveLoop((l16, l17, l18, l19))

s04 = gmsh.model.geo.addPlaneSurface((cl04,))
s05 = gmsh.model.geo.addPlaneSurface((cl05,))
s06 = gmsh.model.geo.addPlaneSurface((cl06,))
s07 = gmsh.model.geo.addPlaneSurface((cl07,))
s08 = gmsh.model.geo.addPlaneSurface((cl08,))

l20 = gmsh.model.geo.addLine(p01d, p02d)
l21 = gmsh.model.geo.addLine(p02d, p03d)
l22 = gmsh.model.geo.addLine(p03d, p04d)
l23 = gmsh.model.geo.addLine(p04d, p01d)
l24 = gmsh.model.geo.addLine(p01, p01d)
l25 = gmsh.model.geo.addLine(p02, p02d)
l26 = gmsh.model.geo.addLine(p03, p03d)
l27 = gmsh.model.geo.addLine(p04, p04d)

cl09 = gmsh.model.geo.addCurveLoop((l24, l20, -l25, -l01))
cl10 = gmsh.model.geo.addCurveLoop((l25, l21, -l26, -l02))
cl11 = gmsh.model.geo.addCurveLoop((l26, l22, -l27, -l03))
cl12 = gmsh.model.geo.addCurveLoop((l27, l23, -l24, -l04))
cl13 = gmsh.model.geo.addCurveLoop((l20, l21, l22, l23))

s09 = gmsh.model.geo.addPlaneSurface((cl09,))
s10 = gmsh.model.geo.addPlaneSurface((cl10,))
s11 = gmsh.model.geo.addPlaneSurface((cl11,))
s12 = gmsh.model.geo.addPlaneSurface((cl12,))
s13 = gmsh.model.geo.addPlaneSurface((cl13,))

sl01 = gmsh.model.geo.addSurfaceLoop((s02, s03, s04, s05, s06, s07, s08))
sl02 = gmsh.model.geo.addSurfaceLoop((s01, s04, s05, s06, s07, s08, s09, s10, s11, s12, s13))

p1c = gmsh.model.geo.addPoint(-5, -5, -5, mesh_size)
p2c = gmsh.model.geo.addPoint(5, -5, -5, mesh_size)
p3c = gmsh.model.geo.addPoint(5, 5, -5, mesh_size)
p4c = gmsh.model.geo.addPoint(-5, 5, -5, mesh_size)
p5c = gmsh.model.geo.addPoint(-5, -5, -15, mesh_size)
p6c = gmsh.model.geo.addPoint(5, -5, -15, mesh_size)
p7c = gmsh.model.geo.addPoint(5, 5, -15, mesh_size)
p8c = gmsh.model.geo.addPoint(-5, 5, -15, mesh_size)

l01c = gmsh.model.geo.addLine(p1c, p2c)
l02c = gmsh.model.geo.addLine(p2c, p3c)
l03c = gmsh.model.geo.addLine(p3c, p4c)
l04c = gmsh.model.geo.addLine(p4c, p1c)

l05c = gmsh.model.geo.addLine(p5c, p6c)
l06c = gmsh.model.geo.addLine(p6c, p7c)
l07c = gmsh.model.geo.addLine(p7c, p8c)
l08c = gmsh.model.geo.addLine(p8c, p5c)

l09c = gmsh.model.geo.addLine(p1c, p5c)
l10c = gmsh.model.geo.addLine(p2c, p6c)
l11c = gmsh.model.geo.addLine(p3c, p7c)
l12c = gmsh.model.geo.addLine(p4c, p8c)

cl14 = gmsh.model.geo.addCurveLoop((l01c, l02c, l03c, l04c))
cl15 = gmsh.model.geo.addCurveLoop((l05c, l06c, l07c, l08c))
cl16 = gmsh.model.geo.addCurveLoop((l01c, l10c, -l05c, -l09c))
cl17 = gmsh.model.geo.addCurveLoop((l02c, l11c, -l06c, -l10c))
cl18 = gmsh.model.geo.addCurveLoop((l03c, l12c, -l07c, -l11c))
cl19 = gmsh.model.geo.addCurveLoop((l04c, l09c, -l08c, -l12c))

s14 = gmsh.model.geo.addPlaneSurface((cl14,))
s15 = gmsh.model.geo.addPlaneSurface((cl15,))
s16 = gmsh.model.geo.addPlaneSurface((cl16,))
s17 = gmsh.model.geo.addPlaneSurface((cl17,))
s18 = gmsh.model.geo.addPlaneSurface((cl18,))
s19 = gmsh.model.geo.addPlaneSurface((cl19,))

sl03 = gmsh.model.geo.addSurfaceLoop((s14, s15, s16, s17, s18, s19))

v01 = gmsh.model.geo.addVolume((sl01, sl03))
v02 = gmsh.model.geo.addVolume((sl02,))
v03 = gmsh.model.geo.addVolume((sl03,))

gmsh.model.geo.synchronize()

gmsh.model.geo.addPhysicalGroup(2, (s01, s02, s03), name='Surface')
gmsh.model.geo.addPhysicalGroup(3, (v01, v02), name='Host')
gmsh.model.geo.addPhysicalGroup(3, (v03,), name='Body')

gmsh.model.geo.synchronize()

gmsh.model.mesh.generate(3)

gmsh.write("topo.vtk")


# %%
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName("topo.vtk")
reader.Update()
mesh = reader.GetOutput()

group = vtk_to_numpy(mesh.GetCellData().GetScalars())
print(np.unique(group))

sigma = np.empty(group.shape, dtype=np.float64)
sigma[group == 2] = 0.001
sigma[group == 3] = 0.1

s = numpy_to_vtk(sigma)
s.SetName("Conductivity")
mesh.GetCellData().AddArray(s)

writer = vtk.vtkXMLUnstructuredGridWriter()
writer.SetFileName("topo.vtu")
writer.SetInputData(mesh)
writer.Write()

# %%
os.remove("topo.vtk")