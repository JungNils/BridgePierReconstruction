import ifcopenshell
from ifcopenshell.api import run
import numpy as np
import plane_util
import util

def main():
    model = ifcopenshell.file(schema="IFC4X3")

    project = run("root.create_entity", model, ifc_class="IfcProject", name="Project")

    length = run("unit.add_si_unit", model, unit_type="LENGTHUNIT")
    run("unit.assign_unit", model, units=[length])

    context = run("context.add_context", model, context_type="Model")

    body = run("context.add_context", model, context_type="Model",
               context_identifier="Body", target_view="MODEL_VIEW", parent=context)

    site = run("root.create_entity", model, ifc_class="IfcSite", name="Site")
    bridge = run("root.create_entity", model, ifc_class="IfcBridge", name="Bridge")

    run("aggregate.assign_object", model, relating_object=project, products=[site])
    run("aggregate.assign_object", model, relating_object=site, products=[bridge])

    return model, body, bridge


def ifc_prismoid(df, dataname):
    df['corners'] = df['corners'].apply(lambda x: [tuple(coord) for coord in x])

    model, body, bridge = main()

    for index, row in df.iterrows():
        vertices = row['corners']

        vertices = [list(map(float, coord)) for coord in vertices]

        print(vertices)

        column = run("root.create_entity", model, ifc_class="IfcColumn")

        faces = [plane_util.generate_faces(vertices)]
        edges = [plane_util.generate_edges(vertices)]
        vertices = [vertices]
        representation = run("geometry.add_mesh_representation", model, context=body, vertices=vertices, faces=faces,
                             edges=edges)

        run("geometry.assign_representation", model, product=column, representation=representation)

        run("aggregate.assign_object", model, relating_object=bridge, products=[column])

    filename = dataname.split(".xyz")[0] + ".ifc"
    output_path = "../RESULTS/" + filename
    model.write(output_path)
    print(f"IFC-Datei {filename} erfolgreich in {output_path} erstellt.")


def ifc_cylinder(df, dataname):
    model, body, bridge = main()

    for index, row in df.iterrows():
        center = np.array([row["center_x"], row["center_y"], row["center_z"]])
        axis_vector = np.array([row["normal_x"], row["normal_y"], row["normal_z"]])
        radius = row["radius"]
        height = row["height"]

        # Vektoren in positive z-Richtung umwandeln
        if axis_vector[2] < 0:
            axis_vector = -axis_vector

        column = run("root.create_entity", model, ifc_class="IfcColumn")

        profile = model.create_entity("IfcCircleProfileDef", ProfileType="AREA", Radius=radius)

        cylinder_representation = run("geometry.add_profile_representation", model, context=body, profile=profile,
                                      depth=height)

        rotation_matrix = util.vector_to_matrix(axis_vector)
        rotation_matrix[:, 3][0:3] = center

        run("geometry.edit_object_placement", model, product=column, matrix=rotation_matrix, is_si=True)

        run("geometry.assign_representation", model, product=column, representation=cylinder_representation)

        run("aggregate.assign_object", model, relating_object=bridge, products=[column])

    filename = dataname.split(".xyz")[0] + ".ifc"
    output_path = "../RESULTS/" + filename
    model.write(output_path)
    print(f"IFC-Datei {filename} erfolgreich in {output_path} erstellt.")

