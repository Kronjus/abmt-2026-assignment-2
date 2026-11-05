from config import MZ_DIR, MATSIM_POPULATION, BOUNDARY_FILE, load_boundary, inside

print("MZ folder exists:        ", MZ_DIR.exists())
print("MATSim population exists:", MATSIM_POPULATION.exists())
print("Boundary file exists:    ", BOUNDARY_FILE.exists())

b = load_boundary()
print("Boundary type:", b.geom_type)
print("Bounds (LV95):", [round(v) for v in b.bounds])
print("Area (km2):   ", round(b.area / 1e6, 1))

# Two test points: Fribourg city centre (should be True), Zurich (should be False)
print("Test points:  ", inside([2578500, 2683000], [1183800, 1248000], b))