import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import random
import pyransac3d as pyrsc
from sklearn.cluster import DBSCAN
import math


class PcProcessing:
    def __init__(self, pcd, nn=16, std_multiplier=8, voxel_size=0.03, nn_distance=0.05, eps=0.5, min_samples=200):
        self.pcd = pcd
        self.nn = nn
        self.std_multiplier = std_multiplier
        self.voxel_size = voxel_size
        self.nn_distance = nn_distance
        self.eps = eps
        self.min_samples = min_samples

    def filter_outliers(self):
        self.filtered_pcd, self.outliers = outlier_filter(self.pcd, nn=self.nn, std_multiplier=self.std_multiplier)
        return self.filtered_pcd, self.outliers

    def downsample(self):
        self.pcd_downsampled = self.filtered_pcd.voxel_down_sample(voxel_size=self.voxel_size)
        return self.pcd_downsampled

    def estimate_normals(self):
        radius_normals = self.nn_distance * 4
        self.pcd_downsampled.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normals, max_nn=self.nn),
            fast_normal_computation=True
        )

    def cluster_dbscan(self):
        self.clustered_pcd, self.labels, self.pcd_arr = dbscan_clustering(self.pcd_downsampled, eps=self.eps,
                                                                          min_samples=self.min_samples)
        return self.clustered_pcd, self.labels, self.pcd_arr

    def process(self):
        self.filter_outliers()
        self.downsample()
        self.cluster_dbscan()
        return self.clustered_pcd, self.labels, self.pcd_arr

    def visualize(self, show_outliers=False):
        if show_outliers:
            o3d.visualization.draw_geometries([
                self.pcd_downsampled.paint_uniform_color([0.6, 0.6, 0.6]),
                self.outliers
            ])
        else:
            o3d.visualization.draw_geometries([
                self.pcd_downsampled.paint_uniform_color([0.6, 0.6, 0.6])
            ])


def outlier_filter(pcd, nn=16, std_multiplier=5):
    filtered_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nn, std_ratio=std_multiplier)
    outliers = pcd.select_by_index(ind, invert=True)
    outliers.paint_uniform_color([1, 0, 0])
    return filtered_pcd, outliers


def dbscan_clustering(pcd, eps=0.5, min_samples=200):
    pcd_arr = np.asarray(pcd.points)

    # DBSCAN-Clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pcd_arr)

    # Cluster-Labels erhalten (für jeden Punkt)
    labels = db.labels_

    # Farben für die Cluster
    colors = plt.get_cmap("tab20", len(set(labels)))

    # Erstellen der Open3D Punktwolke
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_arr)

    # Farben hinzufügen, basierend auf den Cluster-Labels
    colors_array = np.array([colors(label)[:3] if label != -1 else [0, 0, 0] for label in labels])
    pcd.colors = o3d.utility.Vector3dVector(colors_array)

    return pcd, labels, pcd_arr


def calculate_angle(slope):
    """Berechnet den Winkel (in Grad) einer Linie aus der Steigung."""
    return math.degrees(math.atan(slope))


def vector_to_matrix(vector):
    # Normalize the vector
    vector = vector / np.linalg.norm(vector)

    # Use the vector as the Z axis of the local coordinate system
    z_axis = vector

    # Choose an arbitrary vector not parallel to the Z axis for the X axis
    if np.allclose(z_axis, [1, 0, 0]):
        x_axis = np.array([0, 1, 0])
    else:
        x_axis = np.array([1, 0, 0])

    # Compute the local Y axis as the cross product of Z and X
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Recompute the local X axis to ensure orthogonality
    x_axis = np.cross(y_axis, z_axis)

    # Construct the rotation matrix
    rotation_matrix = np.eye(4)
    rotation_matrix[0:3, 0] = x_axis
    rotation_matrix[0:3, 1] = y_axis
    rotation_matrix[0:3, 2] = z_axis

    return rotation_matrix


def find_perpendicular_point(A, d, B):
    # Convert points and vector to numpy arrays for easier calculations
    A = np.array(A)
    d = np.array(d)
    B = np.array(B)

    # Calculate the value of t using the formula derived
    t = -np.dot(d, (A - B)) / np.dot(d, d)

    # Calculate the coordinates of point C
    C = A + t * d

    return C


def check_for_cylinder(pcd_arr, labels):
    # Zufälligen Cluster zur Überprüfung auswählen
    valid_clusters = [cluster_id for cluster_id in set(labels) if cluster_id != -1]
    example_cluster_id = random.choice(valid_clusters)

    # Überprüfen ob Beispielcluster einen Zylinder enthält
    if example_cluster_id is not None:
        # Punkte des Beispielclusters extrahieren
        class_member_mask = (labels == example_cluster_id)
        cluster_points = pcd_arr[class_member_mask]

        # Punktwolke für den Beispielcluster erstellen
        pcd_example_cluster = o3d.geometry.PointCloud()
        pcd_example_cluster.points = o3d.utility.Vector3dVector(cluster_points)

        pcd_arr_example_cluster = np.asarray(pcd_example_cluster.points)
        cylinder = pyrsc.Cylinder()

        # Approximating diameter and calculating z_range
        x_coordinates = pcd_arr_example_cluster[:, 0]
        diameter_approx = np.max(x_coordinates) - np.min(x_coordinates)
        max_tilt = 20  # Maximum tilt of a pier/column in %
        z_range = diameter_approx * (max_tilt / 100)

        # Exec RANSAC
        match = 0
        center, normal, radius, inliers = cylinder.fit(pcd_arr_example_cluster, thresh=0.1, maxIteration=500,
                                                       z_range=z_range)
        plane = pcd_example_cluster.select_by_index(inliers)
        match = len(plane.points) / len(pcd_example_cluster.points) * 100
        print(f"Beispielcluster zur Überprüfung auf zylindrische Stützen {example_cluster_id}: Übereinstimmung {match:.2f}%")

        # Überprüfung der Übereinstimmung
        threshold = 80
        if match < threshold:
            print("Die Punktwolke enthält keine zylindrischen Stützen.")
            return False
        else:
            return True


def project_point_onto_plane(point, plane_point, plane_normal):
    plane_normal = plane_normal / np.linalg.norm(plane_normal)  # Normalisierung des Normalenvektors
    vector_to_plane = point - plane_point
    distance_to_plane = np.dot(vector_to_plane, plane_normal)
    projected_point = point - distance_to_plane * plane_normal
    return projected_point


def vertex_half_circle(all_corners, i, z):
    point1 = np.array(all_corners[i])[:2]
    point2 = np.array(all_corners[i + 2])[:2]

    # Berechnung des Mittelpunkts
    M = (point1 + point2) / 2
    # print("Mittelpunkt:", M)

    # Richtungsvektor zwischen den Punkten
    v = point2 - point1
    # print("Richtungsvektor v:", v)
    x, y = v
    length = math.sqrt(x ** 2 + y ** 2)

    # Normalenvektor (senkrechter Vektor)
    v_normal = np.array([-v[1], v[0]])  # 90 Grad Drehung des Vektors

    # print("v_normal:", v_normal)

    # Normierung des Normalenvektors, damit er die gleiche Länge wie der ursprüngliche Vektor hat
    x, y = v_normal
    length_normal = math.sqrt(x ** 2 + y ** 2)

    if length_normal == 0:
        raise ValueError("Der Vektor kann nicht normiert werden, da die Länge 0 ist.")

    v_normal /= length_normal
    # print("v_normal_norm:", v_normal)
    v_normal *= length / 2

    # Schritt 4: Berechnung der Punkte auf der Senkrechten
    Q1 = M + v_normal
    Q2 = M - v_normal

    vertex1 = np.asarray([Q1[0], Q1[1], z[2]])
    vertex2 = np.asarray([Q2[0], Q2[1], z[2]])

    return vertex1, vertex2
