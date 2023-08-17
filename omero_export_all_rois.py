import numpy as np
import omero_toolbox as omero
from getpass import getpass
import sys
import time
from skimage import draw

total_tic = time.perf_counter()

# Define variables
USER = sys.argv[1]
PASS = sys.argv[2]
GROUP = sys.argv[3]
DATASET = sys.argv[4]
HOST = sys.argv[5]
PORT = sys.argv[6]
ROI_COMMENTS = ('PC', 'PT')
# ROI_COMMENTS = None

print(f'Running on {HOST}')

try:
    # Open the connection to OMERO
    conn = omero.open_connection(username=USER,
                                 password=PASS,
                                 host=HOST,
                                 port=PORT,
                                 group=GROUP,
                                 keep_alive=60)

    dataset_id = int(DATASET)

    roi_filter = ROI_COMMENTS

    dataset = omero.get_dataset(conn, dataset_id)

    project = dataset.getParent()

    new_dataset_name = f'{dataset.getName()}_rois'
    new_dataset_description = f'Source Dataset ID: {dataset.getId()}'
    new_dataset = omero.create_dataset(conn,
                                       name=new_dataset_name,
                                       description=new_dataset_description,
                                       parent_project=project)

    images = dataset.listChildren()

    # Loop through images, get ROIs the intensity values, project and save as .npy
    roi_service = conn.getRoiService()

    print(f'Preparation time: {time.perf_counter() - total_tic:0.4f}')

    for counter, image in enumerate(images):
        result = roi_service.findByImage(image.getId(), None)
        for roi in result.rois:
            roi_tic = time.perf_counter()
            shape = roi.getPrimaryShape()
            try:
                shape_comment = shape.getTextValue()._val
            except AttributeError:
                shape_comment = None
            if roi_filter is not None and shape_comment not in roi_filter:
                continue
            tic = time.perf_counter()
            data = omero.get_shape_intensities(image, shape, zero_edge=True)
            print(f'Get Intensities time: {time.perf_counter() - tic:0.4f}')

            mip_data = data.max(axis=0, keepdims=True)
            aip_data = data.mean(axis=0, keepdims=True)
            aip_data = aip_data.astype(data.dtype)

            new_image_name = f'{image.getName().strip()}_{shape_comment}'

            tic = time.perf_counter()
            omero.create_image_from_numpy_array(connection=conn,
                                                data=mip_data,
                                                image_name=f'{new_image_name}_MIP',
                                                image_description=f'Source Image ID:{image.getId()}',
                                                dataset=new_dataset,
                                                source_image_id=image.getId())
            print(f'Crete Image time: {time.perf_counter() - tic:0.4f}')

            tic = time.perf_counter()
            omero.create_image_from_numpy_array(connection=conn,
                                                data=aip_data,
                                                image_name=f'{new_image_name}_AIP',
                                                image_description=f'Source Image ID:{image.getId()}',
                                                dataset=new_dataset,
                                                source_image_id=image.getId())
            print(f'Crete Image time: {time.perf_counter() - tic:0.4f}')

            print(f'roi processing time: {time.perf_counter() - roi_tic:0.4f}')

        print(f'Processed image {counter}')

finally:
    conn.close()
    print('Done')
    print(f'Total execution time: {time.perf_counter() - total_tic:0.4f}')
