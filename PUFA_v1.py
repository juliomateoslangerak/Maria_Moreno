import numpy as np
import pandas as pd

import omero_toolbox as omero
from getpass import getpass

from skimage.filters import threshold_otsu

FILE_NAME_TOKENS = ["expCond1",
                    "expCond2",
                    "gender",
                    "mouseId",
                    "label",
                    "replica",
                    ]

col_names = FILE_NAME_TOKENS + \
            ["ROI_name",
             "threshold_otzu",
             "min_int",
             "max_int",
             "total_area",
             "above_threshold",
             "density_ratio",
             "Roi_id",
             "Image_id",
             "Dataset_id"]

measurements_df = pd.DataFrame(columns=col_names)

dataset_ids = [int(i) for i in input("Datasets: ").split(",")]

project_id = int(input("Project: "))

try:
    # Open the connection to OMERO
    conn = omero.open_connection(username=input("user: "),
                                 password=getpass("pass: "),
                                 host=input("host: "),
                                 port=int(input("port: ")),
                                 group=input("Group: "),
                                 keep_alive=60)

    project = omero.get_project(conn, project_id)

    for dataset_id in dataset_ids:

        dataset = omero.get_dataset(conn, dataset_id)

        images = dataset.listChildren()

        # Loop through images, get ROIs the intensity values, project and save as .npy
        roi_service = conn.getRoiService()

        for image in images:
            print(f"Analyzing image: {image.getName()}")
            result = roi_service.findByImage(image.getId(), None)
            for roi in result.rois:
                shape = roi.getPrimaryShape()
                shape_comment = shape.getTextValue()._val
                print(f"  Analyzing shape: {shape_comment}")

                data = omero.get_shape_intensities(image, shape, zero_edge=True, zero_value="min")
                threshold = threshold_otsu(data)
                min_int = data.min()
                max_int = data.max()
                total_area = np.count_nonzero(data > min_int)
                above_threshold = np.count_nonzero(data > threshold)
                density_ratio = above_threshold / total_area

                row_data = {}
                for token in FILE_NAME_TOKENS:
                    row_data[token] = image.getName().split("_")[FILE_NAME_TOKENS.index(token)]

                row_data.update({"ROI_name": shape_comment,
                                 "threshold_otzu": threshold,
                                 "min_int": min_int,
                                 "max_int": max_int,
                                 "total_area": total_area,
                                 "above_threshold": above_threshold,
                                 "density_ratio": density_ratio,
                                 "Roi_id": roi.getId(),
                                 "Image_id": image.getId(),
                                 "Dataset_id": dataset_id,
                                 })
                measurements_df = measurements_df.append(row_data, ignore_index=True)

    omero_table = omero.create_annotation_table(conn, "data_table",
                                                column_names=measurements_df.columns.tolist(),
                                                column_descriptions=measurements_df.columns.tolist(),
                                                values=[measurements_df[c].values.tolist() for c in measurements_df.columns],
                                                types=None,
                                                namespace="version_1",
                                                table_description="data_table"
                                                )
    omero.link_annotation(project, omero_table)


finally:
    conn.close()
    print('Done')
