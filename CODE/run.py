import reconstruction
import open3d as o3d
import util
import ifc_creation
import time

start_time = time.time()

# Input-Parameter
# Allgemein
dataname = "B1_Piers.xyz"
concave_shape = False   # Handelt es sich um einen konkaven Pfeilerquerschnitt?
visualize = False  # Punktwolke nach Pre-Processing visualisieren

# Pre-Processing
# Outlier-Filter
nn = 16
std_multiplier = 8
# Downsampling
voxel_size = 0.02
# Clustering der gesamten Punktwolke
eps_pc = 0.5
min_samples_pc = 200

# Plane-Intersection-Parameter für prismoide Pfeiler
# RANSAC-Parameter
min_points_factor_pi = 150  # Anzahl der Gesamtpunkte des Clusters werden mit diesem Faktor dividiert
distance_threshold_pi = 0.01
ransac_n_pi = 3
num_iterations_pi = 10000

# Plane-Clustering (muss nur angegeben werden, wenn concave_shape = True)
eps_plane = 0.12
min_samples_plane = 5

# Parameter für zylindrische Pfeiler
# RANSAC-Parameter
max_tilt_cyl = 15
distance_threshold_cyl = 0.02
num_iterations_cyl = 1000

external_params_pre = {"nn": nn, "std_multiplier": std_multiplier, "voxel_size": voxel_size, "eps": eps_pc,
                       "min_samples": min_samples_pc}

external_params_PI = {"min_points_factor": min_points_factor_pi, "distance_threshold": distance_threshold_pi,
                      "ransac_n": ransac_n_pi, "num_iterations": num_iterations_pi, "eps": eps_plane,
                      "min_samples": min_samples_plane}

external_params_cyl = {"max_tilt": max_tilt_cyl, "distance_threshold": distance_threshold_cyl,
                       "num_iterations": num_iterations_cyl}

external_params_CRE = {"distance_threshold_plane": distance_threshold_pi, "ransac_n_plane": ransac_n_pi,
                       "num_iterations_plane": num_iterations_pi, "max_tilt_cyl": max_tilt_cyl,
                       "distance_threshold_cyl": distance_threshold_cyl, "num_iterations_cyl": num_iterations_cyl}

# Punktwolke einlesen
pcd = o3d.io.read_point_cloud("../DATA/" + dataname)
pcd.paint_uniform_color([0.6, 0.6, 0.6])
pcd_original = pcd

# Pre-Processing
processor = util.PcProcessing(pcd, **external_params_pre)
pcd, labels, pcd_arr = processor.process()
if visualize:
    processor.visualize(show_outliers=False)

pre_time = time.time()
pretime = pre_time - start_time
print(f"Laufzeit PreProcessing: {pretime:.4f} Sekunden")

# Unterscheidung, ob es sich um zylindrische oder prismoide Pfeiler handelt
if util.check_for_cylinder(pcd_arr, labels):
    # Geometrische Rekonstruktion zylindrischer Pfeiler
    df = reconstruction.cylindrical_RANSAC(pcd_arr, labels, **external_params_cyl)

    ifc_creation.ifc_cylinder(df, dataname)

else:
    # Geometrische Rekonstruktion prismoider Pfeiler
    df = reconstruction.prismoid_PI(pcd_arr, labels, concave_shape, **external_params_PI)

    ifc_creation.ifc_prismoid(df, dataname)

end_time = time.time()
runtime = end_time - start_time
print(f"Laufzeit: {runtime:.4f} Sekunden")

