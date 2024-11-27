import util
import plane_util
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyransac3d as pyrsc


def prismoid_PI(pcd_arr, labels, concave_shape, min_points_factor, distance_threshold, ransac_n, num_iterations, eps, min_samples):

    # Anzahl der gefundenen Cluster
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f'Anzahl der gefundenen Cluster: {n_clusters}')

    all_planes = []
    column_params = []

    for cluster_id in range(n_clusters):
        if cluster_id == -1:
            continue  # Skip Outliers

        # Punkte des aktuellen Clusters extrahieren
        class_member_mask = (labels == cluster_id)
        cluster_points = pcd_arr[class_member_mask]

        # Find min z-value
        z_values = cluster_points[:, 2]
        min_z = np.min(z_values)

        # Punktwolke für den aktuellen Cluster erstellen
        pcd_cluster = o3d.geometry.PointCloud()
        pcd_cluster.points = o3d.utility.Vector3dVector(cluster_points)
        pcd_cluster.paint_uniform_color([0.6, 0.6, 0.6])

        # Multi RANSAC for plane detection
        segment_models = {}
        segments = {}
        min_points_threshold = len(pcd_cluster.points) / min_points_factor
        rest = pcd_cluster
        all_plane_eq = []
        inlier_count = []
        i = 0

        while len(rest.points) > min_points_threshold:
            colors = plt.get_cmap("tab20")(i % 20)
            segment_models[i], inliers = rest.segment_plane(
                distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
            segments[i] = rest.select_by_index(inliers)
            segments[i].paint_uniform_color(list(colors[:3]))
            rest = rest.select_by_index(inliers, invert=True)
            [a, b, c, d] = segment_models[i]
            if len(inliers) > min_points_threshold:
                if plane_util.remove_horizontal_planes(a, b, c, angle_threshold=np.pi / 18):
                    print(
                        f"Cluster: {cluster_id}. Plane {i + 1} equation: {a:.5f}x + {b:.5f}y + {c:.5f}z + {d:.5f} = 0")
                    print(f"pass {i + 1} done. {len(rest.points)} left. Min_threshold: {min_points_threshold}")
                    # print(f"pass {i + 1}/{max_plane_idx} done.")

                    all_planes.append(segments[i])
                    all_plane_eq.append((a, b, c, d))
                    inlier_count.append(len(inliers))

                else:
                    print(f"Ebene {i + 1} ist horizontal.")

            else:
                print(f"Minimalanzahl an Punkten unterschritten: {len(inliers)} < {min_points_threshold}")

            i = i + 1

        # Ähnliche Ebenen zusammenfassen
        plane_groups = plane_util.group_similar_planes(all_plane_eq, segments)
        dominant_planes_indices = plane_util.select_dominant_planes(plane_groups, inlier_count)

        # Extrahiere die dominanten Ebenen und ihre Gleichungen
        dominant_planes_eq = [all_plane_eq[idx] for idx in dominant_planes_indices]

        # Extraktion der dominanten Segmente zur späteren Visualisierung
        dominant_segments = [segments[idx] for idx in dominant_planes_indices]

        # Füge die nicht-dominanten Segmente wieder dem Rest hinzu
        non_dominant_indices = [i for i in range(len(segments)) if i not in dominant_planes_indices]
        for idx in non_dominant_indices:
            rest += segments[idx]

        print(dominant_segments)

        # Clustering über die gefundenen Ebenen laufen lassen, damit überschüssige Schnittgeraden vermieden werden
        clustered_segments = []
        clustered_planes_eq = []

        if concave_shape:
            for i, segment in enumerate(dominant_segments):
                plane_pcd, plane_labels, plane_pcd_arr = util.dbscan_clustering(segment, eps=eps, min_samples=min_samples)

                n_plane_clusters = len(set(plane_labels)) - (1 if -1 in labels else 0)
                print(n_plane_clusters)

                for cluster_id in range(n_plane_clusters):
                    if cluster_id == -1:
                        continue  # Skip Outliers

                    # Punkte des aktuellen Clusters extrahieren
                    class_member_mask = (plane_labels == cluster_id)
                    plane_cluster_points = plane_pcd_arr[class_member_mask]

                    if len(plane_cluster_points) > len(segment.points)/10:
                        # Punktwolke für den aktuellen Cluster erstellen
                        plane_pcd_cluster = o3d.geometry.PointCloud()
                        plane_pcd_cluster.points = o3d.utility.Vector3dVector(plane_cluster_points)

                        clustered_segments.append(plane_pcd_cluster)
                        clustered_planes_eq.append(dominant_planes_eq[i])

                #o3d.visualization.draw_geometries([segment])
                #o3d.visualization.draw_geometries([plane_pcd])
                #o3d.visualization.draw_geometries([plane_pcd, pcd_cluster])

            dominant_planes_eq = clustered_planes_eq
            dominant_segments = clustered_segments

        """print(f"Dominant segments: {dominant_segments}")
        print(f"Clustered segments: {clustered_segments}")
        print(f"Clustered segments equations: {clustered_planes_eq}")"""

        print(f"Gleichungen aller gefundenen Ebenen: {all_plane_eq}")
        print(f"Ähnlichen Ebenen: {plane_groups}")
        print(f"Gleichungen aller gefilterten Ebenen: {dominant_planes_eq}")

        all_line_points = []
        all_line_directions = []
        all_corners_top = []
        all_corners_bot = []
        intersecting_planes_idx = []
        points_near_lines = []

        # Errechne die Schnittgeraden
        for i in range(len(dominant_planes_eq)):
            for j in range(i + 1, len(dominant_planes_eq)):
                print(f"Ebenen {i} und {j}")
                plane1 = dominant_planes_eq[i]
                plane2 = dominant_planes_eq[j]

                # o3d.visualization.draw_geometries([dominant_segments[i], dominant_segments[j]])

                # Punkte der beiden Ebenen zusammenfügen
                plane_points1 = np.asarray(dominant_segments[i].points)
                plane_points2 = np.asarray(dominant_segments[j].points)
                # plane_points = np.vstack((points_plane1, points_plane2))

                point, direction, valid_points = plane_util.plane_intersection(plane1, plane2, plane_points1,
                                                                               plane_points2)
                if point is not None:
                    all_line_points.append(point)
                    all_line_directions.append(direction)
                    intersecting_planes_idx.append((i, j))
                    points_near_lines.append(valid_points)


        for i in range(len(all_line_points)):
            point = all_line_points[i]
            direction = all_line_directions[i]
            points_near_line_i = points_near_lines[i]
            z_values = [point[2] for point in points_near_line_i]
            max_z = max(z_values)

            corner_top = plane_util.line_point_z_value(point, direction, max_z)
            corner_bot = plane_util.line_point_z_value(point, direction, min_z)
            all_corners_top.append(corner_top)
            all_corners_bot.append(corner_bot)

        # print(f"Corners Top: {all_corners_top}")
        # print(f"PI idx: {intersecting_planes_idx}")

        # Sort corners
        if concave_shape:
            if plane_util.plane_frequency(intersecting_planes_idx):
                print("Die Punkte mussten über angular_sort sortiert werden, da eine oder mehrere Ebenen mehr als 2 "
                      "Schnittgeraden aufweisen.")
                sorted_corners_bot, sort_index = plane_util.radial_sort(all_corners_bot)
                all_corners_top = np.asarray(all_corners_top)
                sorted_corners_top = all_corners_top[sort_index]
            else:
                sorted_corners_bot, sort_index = plane_util.sort_corners_by_planes(all_corners_bot, intersecting_planes_idx)
                all_corners_top = np.asarray(all_corners_top)
                sorted_corners_top = all_corners_top[sort_index]
        else:
            sorted_corners_bot, sort_index = plane_util.radial_sort(all_corners_bot)
            all_corners_top = np.asarray(all_corners_top)
            sorted_corners_top = all_corners_top[sort_index]

        # print(sorted_corners_bot)
        # print(sorted_corners_top)

        all_corners = np.vstack((all_corners_bot, all_corners_top))
        all_sorted_corners = np.vstack((sorted_corners_bot, sorted_corners_top))
        # print(all_sorted_corners)

        #o3d.visualization.draw_geometries(
            #dominant_segments + [rest.paint_uniform_color([0, 0, 0])])

        corners_pcd = o3d.geometry.PointCloud()
        corners_pcd.points = o3d.utility.Vector3dVector(all_corners)
        corners_pcd.paint_uniform_color([0, 1, 0])

        #o3d.visualization.draw_geometries([pcd_cluster, corners_pcd])

        column_params.append({"cluster_id": cluster_id, "corners": all_sorted_corners})

    # Umwandeln der Daten in ein DataFrame-geeignetes Format
    data = []

    for param in column_params:
        row = {'corners': param['corners'].tolist()}  # converting nparray to list
        data.append(row)

    # Erstellen eines DataFrames
    df = pd.DataFrame(data)

    return df


def cylindrical_RANSAC(pcd_arr, labels, max_tilt, distance_threshold, num_iterations):
    geometries = []  # Initialisierung einer Liste zur Visualisierung
    cyl_params = []  # Initialisierung einer Liste zum Export der Ergebnisse

    # RANSAC für jeden Cluster
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue  # Skip Outliers

        # Punkte des aktuellen Clusters extrahieren
        class_member_mask = (labels == cluster_id)
        cluster_points = pcd_arr[class_member_mask]

        # Punktwolke für den aktuellen Cluster erstellen
        pcd_cluster = o3d.geometry.PointCloud()
        pcd_cluster.points = o3d.utility.Vector3dVector(cluster_points)

        pcd_arr_cluster = np.asarray(pcd_cluster.points)
        cylinder = pyrsc.Cylinder()

        # Approximating diameter and calculating z_range
        x_coordinates = pcd_arr_cluster[:, 0]
        diameter_approx = np.max(x_coordinates) - np.min(x_coordinates)
        z_range = diameter_approx * (max_tilt / 100)

        # Exec RANSAC
        match = 0
        while match < 97:  # Wiederholt RANSAC so lange, bis gewünschte Übereinstimmung erreicht ist
            center_circle, normal, radius, inliers = cylinder.fit(pcd_arr_cluster, thresh=distance_threshold,
                                                                  maxIteration=num_iterations, z_range=z_range)
            plane = pcd_cluster.select_by_index(inliers)
            match = len(plane.points) / len(pcd_cluster.points) * 100
            print(f"Cluster {cluster_id}: Übereinstimmung {match:.2f}%")

        # Using the min and max z-Values to calculate the center points at the top and bottom of the cylinder
        plane_array = np.asarray(plane.points)
        plane_z_values = plane_array[:, 2]
        max_z_index = np.argmax(plane_z_values)
        max_z = plane_array[max_z_index]
        min_z_index = np.argmin(plane_z_values)
        min_z = plane_array[min_z_index]

        # Caclulating the intersection between the cylinder axis and a perpendicular line passing through max and min z
        center_top = util.find_perpendicular_point(center_circle, normal, max_z)
        center_bottom = util.find_perpendicular_point(center_circle, normal, min_z)

        height = np.linalg.norm(center_top - center_bottom)
        center = (center_top + center_bottom) / 2

        # Visualisierung der Punktwolke und des Zylinders
        R = pyrsc.get_rotationMatrix_from_vectors([0, 0, 1], normal)

        plane = pcd_cluster.select_by_index(inliers).paint_uniform_color([0, 0, 1])
        not_plane = pcd_cluster.select_by_index(inliers, invert=True).paint_uniform_color([1, 0, 0])

        mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
        mesh_cylinder.compute_vertex_normals()
        mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
        mesh_cylinder = mesh_cylinder.rotate(R, center=[0, 0, 0])
        mesh_cylinder = mesh_cylinder.translate((center[0], center[1], center[2]))

        # Add to geometries for vis
        geometries.append(plane)
        geometries.append(not_plane)
        geometries.append(mesh_cylinder)

        cyl_params.append(
            {"cluster_id": cluster_id, "center": center_bottom, "normal": normal, "radius": radius, "height": height})

        print(f"Cluster {cluster_id}: Übereinstimmung {match:.2f}%")
        print(f"Zylinderparameter: center_bottom={center_bottom}, normal={normal}, radius={radius}, height={height}")

    o3d.visualization.draw_geometries(geometries)

    # Umwandeln der Daten in ein DataFrame-geeignetes Format
    data = []

    for params in cyl_params:
        row = {"cluster_id": params["cluster_id"],
               "center_x": params["center"][0], "center_y": params["center"][1], "center_z": params["center"][2],
               "normal_x": params["normal"][0], "normal_y": params["normal"][1], "normal_z": params["normal"][2],
               "radius": params["radius"],
               "height": params["height"]}
        data.append(row)

    # Erstellen eines DataFrames
    df = pd.DataFrame(data)

    return df


# Erster Ansatz für Pfeiler mit abgerundeten Enden, noch nicht funktionsfähig
def cuboid_rounded_ends(pcd_arr, labels, distance_threshold_plane, ransac_n_plane, num_iterations_plane, max_tilt_cyl,
                        distance_threshold_cyl, num_iterations_cyl):
    # Anzahl der gefundenen Cluster
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f'Anzahl der gefundenen Cluster: {n_clusters}')

    for cluster_id in range(n_clusters):
        if cluster_id == -1:
            continue  # Skip Outliers

        # Punkte des aktuellen Clusters extrahieren
        class_member_mask = (labels == cluster_id)
        cluster_points = pcd_arr[class_member_mask]

        # Sorting by z-coordinate
        sorted_indices = np.argsort(cluster_points[:, 2])
        pcd_sorted = cluster_points[sorted_indices]

        # Find min and max z-value
        z_values = pcd_sorted[:, 2]
        min_z = np.min(z_values)
        max_z = np.median(z_values[-100:])

        # Punktwolke für den aktuellen Cluster erstellen
        pcd_cluster = o3d.geometry.PointCloud()
        pcd_cluster.points = o3d.utility.Vector3dVector(cluster_points)
        pcd_cluster.paint_uniform_color([0.6, 0.6, 0.6])

        # Multi RANSAC for plane detection
        segment_models = {}
        segments = {}
        max_planes = 2
        rest = pcd_cluster
        all_planes = []
        all_plane_eq = []
        inlier_count = []
        geometries = []
        i = 0

        while i < max_planes:
            colors = plt.get_cmap("tab20")(i % 20)
            segment_models[i], inliers = rest.segment_plane(
                distance_threshold=distance_threshold_plane, ransac_n=ransac_n_plane, num_iterations=num_iterations_plane)
            segments[i] = rest.select_by_index(inliers)
            segments[i].paint_uniform_color(list(colors[:3]))
            rest = rest.select_by_index(inliers, invert=True)
            [a, b, c, d] = segment_models[i]
            print(f"Cluster: {cluster_id}. Plane {i + 1} equation: {a:.5f}x + {b:.5f}y + {c:.5f}z + {d:.5f} = 0")
            all_planes.append(segments[i])
            all_plane_eq.append((a, b, c, d))
            inlier_count.append(len(inliers))

            i = i + 1
        for i in range(max_planes):
            geometries.append(segments[i])
        #o3d.visualization.draw_geometries(
            #[segments[i] for i in range(max_planes)] + [rest.paint_uniform_color([0, 0, 0])])

        # Second DBSCAN to separate the two half cylinders
        rest_pcd, labels_rest, rest_pcd_arr = util.dbscan_clustering(rest, eps=0.5, min_samples=100)

        n_clusters_rest = len(set(labels_rest)) - (1 if -1 in labels else 0)
        print(f'Anzahl der gefundenen Cluster: {n_clusters_rest}')

        #o3d.visualization.draw_geometries([rest_pcd])

        all_centers_cyl = []
        all_normals_cyl = []

        for cluster_id_rest in range(n_clusters_rest):
            if cluster_id_rest == -1:
                continue  # Skip Outliers

            # Punkte des aktuellen Clusters extrahieren
            class_member_mask = (labels_rest == cluster_id_rest)
            cluster_points_rest = rest_pcd_arr[class_member_mask]

            rest_pcd_cluster = o3d.geometry.PointCloud()
            rest_pcd_cluster.points = o3d.utility.Vector3dVector(cluster_points_rest)

            # Finding cylinders
            cylinder = pyrsc.Cylinder()

            # Approximating diameter and calculating z_range
            x_coordinates = cluster_points_rest[:, 0]
            diameter_approx = np.max(x_coordinates) - np.min(x_coordinates)
            z_range = diameter_approx * (max_tilt_cyl / 100)

            # Exec RANSAC
            match = 0
            while match < 95:  # Wiederholt RANSAC so lange, bis gewünschte Übereinstimmung erreicht ist
                center_circle, normal, radius, inliers = cylinder.fit(cluster_points_rest, thresh=distance_threshold_cyl,
                                                                      maxIteration=num_iterations_cyl, z_range=z_range)
                plane = rest_pcd_cluster.select_by_index(inliers)
                match = len(plane.points) / len(rest_pcd_cluster.points) * 100
                print(f"Cluster {cluster_id_rest}: Übereinstimmung {match:.2f}%")

            all_normals_cyl.append(normal)

            # Using the min and max z-Values to calculate the center points at the top and bottom of the cylinder
            plane_array = np.asarray(plane.points)
            plane_z_values = plane_array[:, 2]
            max_z_index = np.argmax(plane_z_values)
            max_z = plane_array[max_z_index]
            min_z_index = np.argmin(plane_z_values)
            min_z = plane_array[min_z_index]

            # Caclulating the intersection between the cylinder axis and a perpendicular line passing through max and min z
            center_top = util.find_perpendicular_point(center_circle, normal, max_z)
            center_bottom = util.find_perpendicular_point(center_circle, normal, min_z)

            height = np.linalg.norm(center_top - center_bottom)
            center = (center_top + center_bottom) / 2

            all_centers_cyl.append(center)

            # Visualisierung der Punktwolke und des Zylinders
            R = pyrsc.get_rotationMatrix_from_vectors([0, 0, 1], normal)

            plane = rest_pcd_cluster.select_by_index(inliers).paint_uniform_color([0, 1, 0])
            not_plane = rest_pcd_cluster.select_by_index(inliers, invert=True)

            mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
            mesh_cylinder.compute_vertex_normals()
            mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
            mesh_cylinder = mesh_cylinder.rotate(R, center=[0, 0, 0])
            mesh_cylinder = mesh_cylinder.translate((center[0], center[1], center[2]))

            # Add to geometries for vis
            geometries.append(plane)
            geometries.append(not_plane)
            geometries.append(mesh_cylinder)

            print(f"Cluster {cluster_id_rest}: Übereinstimmung {match:.2f}%")
            print(f"Zylinderparameter: center={center}, normal={normal}, radius={radius}, height={height}")

        o3d.visualization.draw_geometries(geometries)

        # Übergangsgeraden ermitteln
        # Ebenenparameter
        all_corners_top = []
        all_corners_bot = []
        i = 0
        while i < 2:
            j = 0
            while j < 2:
                params = all_plane_eq[i]
                plane_normal = np.array(params[:3])
                plane_point = np.array([0, 0-params[3], 0])
                # print(plane_normal, plane_point)

                # Zylinderparameter
                line_point = np.array(all_centers_cyl[j])
                line_direction = np.array(all_normals_cyl[j])
                # print(line_point, line_direction)

                projected_point = util.project_point_onto_plane(line_point, plane_point, plane_normal)
                projected_normal = line_direction - np.dot(line_direction, plane_normal) * plane_normal

                # print(projected_point, projected_normal)

                point_top = plane_util.line_point_z_value(projected_point, projected_normal, max_z[2])
                point_bot = plane_util.line_point_z_value(projected_point, projected_normal, min_z[2])

                all_corners_top.append(point_top)
                all_corners_bot.append(point_bot)

                j += 1
            i += 1

        # Scheitelpunkte der Zylinder finden
        # Top corners
        i = 0
        while i < 2:
            vertex_top1, vertex_top2 = util.vertex_half_circle(all_corners_top, i, max_z)
            all_corners_top.append(vertex_top1)
            all_corners_top.append(vertex_top2)

            vertex_bot1, vertex_bot2 = util.vertex_half_circle(all_corners_bot, i, min_z)
            all_corners_bot.append(vertex_bot1)
            all_corners_bot.append(vertex_bot2)

            i += 1

        corners_pcd = o3d.geometry.PointCloud()
        corners_pcd.points = o3d.utility.Vector3dVector(all_corners_top)
        corners_pcd.paint_uniform_color([0, 1, 0])

        o3d.visualization.draw_geometries([corners_pcd, pcd_cluster])


