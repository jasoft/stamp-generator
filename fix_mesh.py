#!/usr/bin/env python3
"""Fix and optimize the stamp STL mesh."""
import trimesh
import numpy as np
import os

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
STL_PATH = os.path.join(WORK_DIR, "公章_40mm.stl")

print("Loading mesh...")
mesh = trimesh.load(STL_PATH)
print(f"Before: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces, watertight={mesh.is_watertight}")

# Remove degenerate faces (zero-area triangles)
faces = mesh.faces
v0 = mesh.vertices[faces[:, 0]]
v1 = mesh.vertices[faces[:, 1]]
v2 = mesh.vertices[faces[:, 2]]
cross = np.cross(v1 - v0, v2 - v0)
areas = np.linalg.norm(cross, axis=1)
valid = areas > 1e-12
mesh.faces = faces[valid]

# Cleanup
mesh.merge_vertices()
mesh.process()
trimesh.repair.fill_holes(mesh)

print(f"After repair: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces, watertight={mesh.is_watertight}")

# Simplify to reduce file size
target = 150000
if len(mesh.faces) > target:
    print(f"Simplifying to ~{target:,} faces...")
    mesh = mesh.simplify_quadric_decimation(target)
    mesh.merge_vertices()
    mesh.process()
    print(f"After: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces, watertight={mesh.is_watertight}")

# Re-export
mesh.export(STL_PATH)
print(f"\nFinal STL: {STL_PATH}")
print(f"  Vertices: {len(mesh.vertices):,}")
print(f"  Triangles: {len(mesh.faces):,}")
print(f"  Dimensions: {mesh.extents[0]:.1f} x {mesh.extents[1]:.1f} x {mesh.extents[2]:.1f} mm")
print(f"  File size: {os.path.getsize(STL_PATH) / 1e6:.1f} MB")
print(f"  Watertight: {mesh.is_watertight}")
