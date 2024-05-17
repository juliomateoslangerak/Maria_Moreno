import numpy as np
# import omero_rois
import pandas as pd

from skimage.filters import apply_hysteresis_threshold
from skimage.restoration import rolling_ball
from skimage.measure import label

import omero_toolbox
import ezomero
from getpass import getpass

THRESHOLD_MAX = 180
THRESHOLD_MIN = 120
MIN_DISTANCE = 2
ROLLING_BALL_RADIUS = 50

# FILE_NAME_TOKENS = ["exp_model",
#                     "mouse_id",
#                     "genotype",
#                     "labelling",
#                     ]
FILE_NAME_TOKENS = ["exp_model",
                    "labelling",
                    ]

col_names = FILE_NAME_TOKENS + \
            ["ROI_name",
             "threshold_max",
             "threshold_min"
             "rolling_ball_size",
             "min_int",
             "max_int",
             "total_area",
             "segmented_area",
             "segmented_area_ratio",
             "total_raw_intensity",
             "total_raw_intensity",
             "segmented_raw_intensity",
             "total_rolling_ball_intensity",
             "segmented_rolling_ball_intensity",
             "roi_id",
             "image_id",
             "prob_image_id",
             "dataset_id",
             ]

measurements = []

dataset_id = int(input("Dataset: ") or 23775)
prob_dataset_id = int(input("Probabilities dataset: ") or 24713)

try:
    # Open the connection to OMERO
    conn = omero_toolbox.open_connection(username=str(input("user: ") or "mateos"),
                                         password=getpass("pass: "),
                                         host=str(input("host: ") or "omero.mri.cnrs.fr"),
                                         port=int(input("port: ") or 4064),
                                         group=str(input("Group: ") or "novoDA"),
                                         keep_alive=60)

    dataset = omero_toolbox.get_dataset(conn, dataset_id)
    prob_dataset = omero_toolbox.get_dataset(conn, prob_dataset_id)

    print(f"Analyzing dataset: {dataset.getName()}")

    project = dataset.getParent()
    image_ids = ezomero.get_image_ids(conn, dataset=dataset_id)
    images = {conn.getObject("Image", i).get_name(): i for i in image_ids}
    prob_image_ids = ezomero.get_image_ids(conn, dataset=prob_dataset_id)
    prob_images = {conn.getObject("Image", i).get_name(): i for i in prob_image_ids}

    roi_service = conn.getRoiService()

    for image_name in images.keys():
        print(f"Analyzing image: {image_name}")
        raw_image = conn.getObject("Image", images[image_name])
        if "label image" in raw_image.getName() or "macro image" in raw_image.getName():
            continue

        try:
            prob_image = conn.getObject("Image", prob_images[f"{image_name}_PROB"])
        except KeyError:
            continue

        result = roi_service.findByImage(images[image_name], None)
        for roi in result.rois:
            shape = roi.getPrimaryShape()
            shape_comment = shape.getTextValue()._val
            print(f"  Analyzing shape: {shape_comment}")

            raw_data = omero_toolbox.get_shape_intensities(raw_image, shape, zero_edge=True, zero_value="zero")
            raw_data = raw_data[0, 1, 0,...]
            prob_data = omero_toolbox.get_shape_intensities(prob_image, shape, zero_edge=True, zero_value="min")
            prob_data = prob_data[0, 0, 0,...]

            thresholded = apply_hysteresis_threshold(prob_data,
                                                     low=THRESHOLD_MAX,
                                                     high=THRESHOLD_MIN
                                                     )

            background = rolling_ball(raw_data, radius=ROLLING_BALL_RADIUS, num_threads=8)
            raw_data_foreground = raw_data - background

            # labels = label(thresholded)
            # labels = label(labels > 0)
            #
            # masks = omero_rois.masks_from_label_image(labelim=labels, rgba=(0, 255, 0, 120), text=shape_comment)

            # mask = labels > 0
            #
            # points = [tuple(float(c) for c in p.split(',')) for p in shape.getPoints()._val.split()]
            # x_pos = min([int(x) for x, _ in points])
            # y_pos = min([int(y) for _, y in points])
            #
            # mask = omero_toolbox.create_shape_mask(np.transpose(thresholded), x_pos=x_pos, y_pos=y_pos,
            #                                        z_pos=None, t_pos=None, mask_name=shape_comment)
            # omero_toolbox.create_roi(conn, raw_image, [mask])
            # omero_toolbox.create_roi(conn, raw_image, masks)

            min_int = np.min(raw_data[np.nonzero(raw_data)])
            max_int = raw_data.max()
            total_area = np.count_nonzero(raw_data > 0)
            segmented_area = np.count_nonzero(thresholded > 0)
            segmented_area_ratio = segmented_area / total_area
            total_raw_intensity = np.sum(raw_data)
            segmented_raw_intensity = np.sum(raw_data * thresholded)
            total_rolling_ball_intensity = np.sum(raw_data_foreground)
            segmented_rolling_ball_intensity = np.sum(raw_data_foreground * thresholded)

            row_data = {}
            for token in FILE_NAME_TOKENS:
                row_data[token] = image_name.split("-")[FILE_NAME_TOKENS.index(token)]

            row_data.update({"ROI_name": shape_comment,
                             "threshold_max": THRESHOLD_MAX,
                             "threshold_min": THRESHOLD_MIN,
                             "rolling_ball_size": ROLLING_BALL_RADIUS,
                             "min_int": min_int,
                             "max_int": max_int,
                             "total_area": total_area,
                             "segmented_area": segmented_area,
                             "segmented_area_ratio": segmented_area_ratio,
                             "total_raw_intensity": total_raw_intensity,
                             "segmented_raw_intensity": segmented_raw_intensity,
                             "total_rolling_ball_intensity": total_rolling_ball_intensity,
                             "segmented_rolling_ball_intensity": segmented_rolling_ball_intensity,
                             "roi_id": roi.getId().getValue(),
                             "image_id": raw_image.getId(),
                             "prob_image_id": prob_image.getId(),
                             "dataset_id": dataset.getId(),
                             })
            measurements.append(row_data)

    measurements_df = pd.DataFrame.from_records(measurements)
    measurements_df.to_csv(f"/home/julio/PycharmProjects/Maria_Moreno/novoDA_dataset-{dataset_id}_v4.csv", index=False)
    omero_table = omero_toolbox.create_annotation_table(
        conn, "data_table",
        column_names=measurements_df.columns.tolist(),
        column_descriptions=measurements_df.columns.tolist(),
        values=[measurements_df[c].values.tolist() for c in measurements_df.columns],
        types=None,
        namespace="version_1",
        table_description="data_table"
        )
    omero_toolbox.link_annotation(project, omero_table)
except Exception as e:
    print(e)

finally:
    conn.close()
    print('Done')
