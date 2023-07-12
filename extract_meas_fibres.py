import numpy as np
import pandas as pd

from skimage.filters import apply_hysteresis_threshold
from skimage.measure import label

import omero_toolbox
import ezomero
from getpass import getpass

THRESHOLD_MAX = 180
THRESHOLD_MIN = 120
MIN_DISTANCE = 2

FILE_NAME_TOKENS = ["expCond1",
                    "expCond2",
                    "gender",
                    "mouseId",
                    "label",
                    "replica",
                    ]

col_names = FILE_NAME_TOKENS + \
            ["ROI_name",
             "threshold_max",
             "threshold_min"
             "min_int",
             "max_int",
             "total_area",
             "above_threshold",
             "density_ratio",
             "roi_id",
             "image_id",
             "prob_image_id",
             "dataset_id",
             ]

measurements_df = pd.DataFrame(columns=col_names)

dataset_id = int(input("Dataset: "))

try:
    # Open the connection to OMERO
    conn = omero_toolbox.open_connection(username=input("user: "),
                                 password=getpass("pass: "),
                                 host=str(input("host: ") or "omero.mri.cnrs.fr"),
                                 port=int(input("port: ") or 4064),
                                 group=input("Group: "),
                                 keep_alive=60)

    dataset = omero_toolbox.get_dataset(conn, dataset_id)

    print(f"Analyzing dataset: {dataset.getName()}")

    project = dataset.getParent()
    image_ids = ezomero.get_image_ids(conn, dataset=dataset_id)
    images = {conn.getObject("Image", i).get_name(): conn.getObject("Image", i) for i in image_ids}

    roi_service = conn.getRoiService()

    for image_name, raw_image in images.items():
        print(f"Analyzing image: {image_name}")
        if image_name.endswith("_PROB"):
            continue
        prob_image = images[f"{image_name}_PROB"]
        result = roi_service.findByImage(raw_image.getId(), None)
        for roi in result.rois:
            shape = roi.getPrimaryShape()
            shape_comment = shape.getTextValue()._val
            print(f"  Analyzing shape: {shape_comment}")

            raw_data = omero_toolbox.get_shape_intensities(raw_image, shape, zero_edge=True, zero_value="zero")
            prob_data = omero_toolbox.get_shape_intensities(prob_image, shape, zero_edge=True, zero_value="min")

            thresholded = apply_hysteresis_threshold(prob_data[0, 0, 0,...],
                                                     low=THRESHOLD_MAX,
                                                     high=THRESHOLD_MIN
                                                     )

            labels = label(thresholded)

            mask = labels > 0

            points = [tuple(float(c) for c in p.split(',')) for p in shape.getPoints()._val.split()]
            x_pos = min([int(x) for x, _ in points])
            y_pos = min([int(y) for _, y in points])

            mask = omero_toolbox.create_shape_mask(np.transpose(mask), x_pos=x_pos, y_pos=y_pos,
                                                   z_pos=None, t_pos=None, mask_name=shape_comment)
            omero_toolbox.create_roi(conn, raw_image, [mask])

            min_int = np.min(raw_data[np.nonzero(raw_data)])
            max_int = raw_data.max()
            total_area = np.count_nonzero(raw_data > 0)
            above_threshold = np.count_nonzero(labels > 0)
            density_ratio = above_threshold / total_area

            row_data = {}
            for token in FILE_NAME_TOKENS:
                row_data[token] = image_name.split("_")[FILE_NAME_TOKENS.index(token)]

            row_data.update({"ROI_name": shape_comment,
                             "threshold_max": THRESHOLD_MAX,
                             "threshold_min": THRESHOLD_MIN,
                             "min_int": min_int,
                             "max_int": max_int,
                             "total_area": total_area,
                             "above_threshold": above_threshold,
                             "density_ratio": density_ratio,
                             "roi_id": roi.getId().getValue(),
                             "image_id": raw_image.getId(),
                             "prob_image_id": prob_image.getId(),
                             "dataset_id": dataset.getId(),
                             })
            measurements_df = measurements_df.append(row_data, ignore_index=True)

    measurements_df.to_csv(f"PUFA_dataset-{dataset_id}_v2.csv", index=False)
    # omero_table = omero_toolbox.create_annotation_table(conn, "data_table",
    #                                             column_names=measurements_df.columns.tolist(),
    #                                             column_descriptions=measurements_df.columns.tolist(),
    #                                             values=[measurements_df[c].values.tolist() for c in measurements_df.columns],
    #                                             types=None,
    #                                             namespace="version_1",
    #                                             table_description="data_table"
    #                                             )
    # omero_toolbox.link_annotation(project, omero_table)
except Exception as e:
    print(e)

finally:
    conn.close()
    print('Done')
