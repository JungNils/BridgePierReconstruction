import numpy as np
from sympy import symbols, Eq, solve
import math


def plane_intersection(plane1, plane2, plane_points1, plane_points2, angle_threshold=np.pi / 18):
    a1, b1, c1, d1 = plane1
    a2, b2, c2, d2 = plane2

    # Symbole für die Unbekannten
    x, y, z = symbols('x y z')

    # Ebenengleichungen aufstellen
    eq1 = Eq(a1 * x + b1 * y + c1 * z + d1, 0)
    eq2 = Eq(a2 * x + b2 * y + c2 * z + d2, 0)

    # Normalenvektoren der Ebenen berechnen
    normal1 = np.array([a1, b1, c1])
    normal1 = normal1 / np.linalg.norm(normal1)
    normal2 = np.array([a2, b2, c2])
    normal2 = normal2 / np.linalg.norm(normal2)
    direction = np.cross(normal1, normal2)
    angle = np.arccos(np.clip(np.dot(normal1, normal2), -1.0, 1.0))

    if angle < angle_threshold:
        print("Die Ebenen sind annähernd parallel, keine eindeutige Schnittgerade.")
        return None, None, None
    else:
        # Einen Punkt auf der Schnittgeraden finden, indem eine der Variablen auf 0 gesetzt wird
        point = None
        if direction[2] != 0:
            # Setze z = 0 und löse nach x und y
            solution = solve((eq1.subs(z, 0), eq2.subs(z, 0)), (x, y))
            if solution:
                point = np.array([solution[x], solution[y], 0])
        elif direction[1] != 0:
            # Setze y = 0 und löse nach x und z
            solution = solve((eq1.subs(y, 0), eq2.subs(y, 0)), (x, z))
            if solution:
                point = np.array([solution[x], 0, solution[z]])
        else:
            # Setze x = 0 und löse nach y und z
            solution = solve((eq1.subs(x, 0), eq2.subs(x, 0)), (y, z))
            if solution:
                point = np.array([0, solution[y], solution[z]])

        if point is not None:
            valid_points = is_valid_line(point, direction, plane_points1, plane_points2)
            if valid_points:
                print(f"Ein Punkt auf der Schnittgerade: {point}")
                print(f"Richtungsvektor der Schnittgerade: {direction}")
                print(f"Geradengleichung in Parameterform: r(t) = {point} + t * {direction}")
                return point, direction, valid_points
            else:
                print(f"Kein Punkt der Punktwolke liegt in der Nähe der Schnittgeraden.")
                return None, None, None
        else:
            print("Konnte keinen Punkt auf der Schnittgeraden finden.")
            return None, None, None


def point_to_line_distance(point, line_point, line_direction):
    point = np.array(point, dtype=np.float64)
    line_point = np.array(line_point, dtype=np.float64)
    line_direction = np.array(line_direction, dtype=np.float64)
    distance = np.linalg.norm(np.cross(line_direction, line_point - point)) / np.linalg.norm(line_direction)
    return distance


def is_valid_line(line_point, line_direction, plane_points1, plane_points2, radius=0.03, height_radius=0.1):
    valid_points = []  # Liste zum Speichern der validen Punkte
    close_points = []
    close_points_control = []
    point_cloud1 = [point.tolist() for point in plane_points1]
    point_cloud2 = [point.tolist() for point in plane_points2]
    for point in point_cloud1:
        distance = point_to_line_distance(point, line_point, line_direction)
        if distance <= height_radius:
            valid_points.append(point)  # Füge den validen Punkt zur Liste hinzu
            if distance <= radius:
                close_points.append(point)

    if len(close_points) == 0:
        valid_points = []
        return valid_points

    for point in point_cloud2:
        distance = point_to_line_distance(point, line_point, line_direction)
        if distance <= height_radius:
            valid_points.append(point)  # Füge den validen Punkt zur Liste hinzu
            if distance <= radius:
                close_points.append(point)
                close_points_control.append(point)

    if len(close_points_control) == 0:
        valid_points = []
        return valid_points

    return valid_points


def line_point_z_value(point, direction, desired_z):
    t = symbols('t')

    # Parametrische Gleichungen der Geraden
    x_param = point[0] + t * direction[0]
    y_param = point[1] + t * direction[1]
    z_param = point[2] + t * direction[2]

    # Gleichung aufstellen: z = desired_z
    eq = Eq(z_param, desired_z)

    # Parameter t berechnen
    solution = solve(eq, t)

    if solution:
        t_value = solution[0]
        # Punkt auf der Geraden berechnen
        x_value = x_param.subs(t, t_value)
        y_value = y_param.subs(t, t_value)
        z_value = z_param.subs(t, t_value)

        print(f"Der Punkt auf der Geraden mit z = {desired_z} ist:")
        print(f"({x_value}, {y_value}, {z_value})")
        line_point = np.array([x_value, y_value, z_value])
        return line_point
    else:
        print(f"Keine Lösung gefunden für z = {desired_z}.")
        return None


def plane_frequency(list):
    flat_list = [element for tupel in list for element in tupel]
    frequency = {}

    # Häufigkeit der Werte zählen
    for element in flat_list:
        if element in frequency:
            frequency[element] += 1
        else:
            frequency[element] = 1

    # Überprüfen, ob irgendein Wert mehr als 2-mal vorkommt
    more_than_twice = any(count > 2 for count in frequency.values())

    if more_than_twice:
        return True
    else:
        return False


def radial_sort(corners):
    corners = np.array(corners)

    # Only use the x- and y-coordinates
    xy_points = corners[:, :2]
    z_points = corners[:, 2:]

    # Compute means for base point
    mx = np.mean(xy_points[:, 0])
    my = np.mean(xy_points[:, 1])

    # compute angles relative to the base point
    angles = []
    for i, (x, y) in enumerate(xy_points):
        angle = math.atan2(y - my, x - mx)
        angles.append((angle, (x, y, z_points[i]), i))

    # sort corners according to their angles
    angles.sort()

    # extract sorted corners
    sorted_corners = [corner for angle, corner, index in angles]
    sorted_indices = [index for angle, corner, index in angles]

    sorted_corners = [list(corner[:2]) + list(corner[2]) for corner in sorted_corners]
    sorted_corners = np.asarray(sorted_corners)

    return sorted_corners, sorted_indices


def sort_corners_by_planes(corners, intersecting_planes_idx):

    sorted_corners = [corners[0]]  # Beginne mit corner 0
    sorted_indices = [0]  # Speichert die Sortierindizes
    used_indices = {0}  # Set für die Indizes der bereits sortierten corners
    current_planes = set(intersecting_planes_idx[0])  # Aktuelle Ebenen (von corner 0)

    # Solange noch Ecken übrig sind
    while len(sorted_corners) < len(corners):
        for i, planes in enumerate(intersecting_planes_idx):
            # Überspringe bereits verwendete Ecken
            if i in used_indices:
                continue

            # Überprüfe, ob eine der aktuellen Ebenen in den neuen Ebenen enthalten ist
            if current_planes & set(planes):  # Schnittmenge der Ebenen
                sorted_corners.append(corners[i])  # Füge die nächste Ecke hinzu
                sorted_indices.append(i)  # Speichere den Index
                used_indices.add(i)  # Markiere die Ecke als verwendet
                current_planes = set(planes)  # Setze die neuen Ebenen
                break

    return sorted_corners, sorted_indices


def generate_faces(vertices):
    num_vertices = len(vertices)

    if num_vertices % 2 != 0:
        raise ValueError("The number of vertices should be even.")

    num_points_per_level = num_vertices // 2

    faces = []

    # Bottom face (connect vertices in the lower level)
    bottom_face = tuple(range(num_points_per_level))
    faces.append(bottom_face)

    # Top face (connect vertices in the upper level)
    top_face = tuple(range(num_points_per_level, num_vertices))
    faces.append(top_face)

    # Side faces (connect each vertex in the lower level to the corresponding vertex in the upper level)
    for i in range(num_points_per_level):
        lower_vertex = i
        upper_vertex = i + num_points_per_level
        next_lower_vertex = (i + 1) % num_points_per_level
        next_upper_vertex = (i + 1) % num_points_per_level + num_points_per_level
        faces.append((lower_vertex, next_lower_vertex, next_upper_vertex, upper_vertex))

    return faces


def generate_edges(vertices):
    num_vertices = len(vertices)

    if num_vertices % 2 != 0:
        raise ValueError("The number of vertices should be even.")

    num_points_per_level = num_vertices // 2

    edges = []

    # Edges on the bottom face (connect consecutive vertices on the lower level)
    for i in range(num_points_per_level):
        next_lower_vertex = (i + 1) % num_points_per_level
        edges.append((i, next_lower_vertex))

    # Edges on the top face (connect consecutive vertices on the upper level)
    for i in range(num_points_per_level):
        current_upper_vertex = i + num_points_per_level
        next_upper_vertex = (i + 1) % num_points_per_level + num_points_per_level
        edges.append((current_upper_vertex, next_upper_vertex))

    # Vertical edges (connect each lower vertex with the corresponding upper vertex)
    for i in range(num_points_per_level):
        edges.append((i, i + num_points_per_level))

    return edges


def are_planes_similar(plane1, plane2, points1, points2, angle_threshold=np.pi / 18, point_distance_threshold=0.05):
    # Extrahiere die Normalenvektoren
    normal1 = np.array(plane1[:3])
    normal2 = np.array(plane2[:3])

    # Berechne den Winkel zwischen den Normalenvektoren
    angle = np.arccos(np.dot(normal1, normal2) / (np.linalg.norm(normal1) * np.linalg.norm(normal2)))

    # Wenn der Winkel größer als der Schwellenwert ist, sind die Ebenen nicht ähnlich
    if angle >= angle_threshold:
        return False

    # Wähle das Segment mit weniger Punkten für die Distanzberechnung
    if len(points1) < len(points2):
        points_to_check = points1
        normal_to_check_against = normal2
        d_value = plane2[3]
    else:
        points_to_check = points2
        normal_to_check_against = normal1
        d_value = plane1[3]

    # Berechne den mittleren Abstand der Punkte zur anderen Plane
    distances = np.abs(np.dot(points_to_check, normal_to_check_against) + d_value) / np.linalg.norm(normal_to_check_against)
    mean_distance = np.mean(distances)
    print(f"Distances are_planes_similar: {distances[:30]}")
    print(f"Mean Distance: {mean_distance}")

    # Überprüfe, ob der mittlere Abstand unter dem Threshold liegt
    return mean_distance < point_distance_threshold


def group_similar_planes(all_plane_eq, segments, angle_threshold=np.pi / 18, point_distance_threshold=0.05):
    plane_groups = []
    used_planes = set()

    for i in range(len(all_plane_eq)):
        if i in used_planes:
            continue

        current_group = [i]
        for j in range(i + 1, len(all_plane_eq)):
            if j in used_planes:
                continue

            points1 = np.asarray(segments[i].points)
            points2 = np.asarray(segments[j].points)

            if are_planes_similar(all_plane_eq[i], all_plane_eq[j], points1, points2, angle_threshold, point_distance_threshold=point_distance_threshold):
                current_group.append(j)
                used_planes.add(j)

        plane_groups.append(current_group)

    return plane_groups


def select_dominant_planes(plane_groups, inlier_count):
    dominant_planes = []

    for group in plane_groups:
        # Finde die Ebene mit der höchsten Anzahl an Inliers
        max_inliers = max(group, key=lambda idx: inlier_count[idx])
        dominant_planes.append(max_inliers)

    return dominant_planes


def remove_horizontal_planes(a, b, c, angle_threshold=np.pi / 18):
    normal = [a, b, c]
    z_axis = [0, 0, 1]

    angle = np.arccos(np.dot(normal, z_axis) / (np.linalg.norm(normal) * np.linalg.norm(z_axis)))

    if angle >= angle_threshold:
        return True
    else:
        return False

