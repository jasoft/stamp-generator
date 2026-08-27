#!/usr/bin/env python3
"""Check watertight status and simplify the stamp STL."""
import trimesh
import numpy as np
import fast_simplification
import os

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
STL_PATH = os.path.join(WORK_DIR, "公章_40mm.stl")

print("Loading mesh...")
mesh = trimesh.load(STL_PATH)
print(f"Original: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")
print(f"  Watertight: {mesh.is_watertight}")
print(f"  Dimensions: {mesh.extents}")

# Simplify using fast_simplification
print("\nSimplifying to ~80k faces...")
v = np.asarray(mesh.vertices, dtype=np.float32)
f = np.asarray(mesh.faces, dtype=np.int32)
v_out, f_out = fast_simplification.simplify(v, f, target_count=80000)

simplified = trimesh.Trimesh(vertices=v_out, faces=f_out)
simplified.merge_vertices()
simplified.process()

print(f"Simplified: {len(simplified.vertices):,} verts, {len(simplified.faces):,} faces")
print(f"  Watertight: {simplified.is_watertight}")
print(f"  Dimensions: {simplified.extents}")

simplified.export(STL_PATH)
print(f"\nFinal STL: {STL_PATH}")
print(f"  File size: {os.path.getsize(STL_PATH) / 1e6:.1f} MB")
