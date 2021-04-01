import numpy as np
import omero_toolbox as omero
from getpass import getpass
from skimage import draw


# Define variables
HOST = 'omero.mri.cnrs.fr'
PORT = 4064
ROI_COMMENTS = ('PC', 'PT')
# ROI_COMMENTS = None


try:
    # Open the connection to OMERO
    conn = omero.open_connection(username=input("Username: "),
                                 password=getpass("OMERO Password: ", None),
                                 host=str(input('server (omero.mri.cnrs.fr): ') or HOST),
                                 port=int(input('port (4064): ') or PORT),
                                 group=input("Group: "))

    dataset_id = int(input('Dataset ID: '))

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

    for counter, image in enumerate(images):
        result = roi_service.findByImage(image.getId(), None)
        for roi in result.rois:
            shape = roi.getPrimaryShape()
            try:
                shape_comment = shape.getTextValue()._val
            except AttributeError:
                shape_comment = None
            if roi_filter is not None and shape_comment not in roi_filter:
                continue
            data = omero.get_shape_intensities(image, shape, zero_edge=True)
            mip_data = data.max(axis=0, keepdims=True)
            aip_data = data.mean(axis=0, keepdims=True)
            aip_data = aip_data.astype(data.dtype)

            new_image_name = f'{image.getName().strip()}_{shape_comment}'
            omero.create_image_from_numpy_array(connection=conn,
                                                data=mip_data,
                                                image_name=f'{new_image_name}_MIP',
                                                image_description=f'Source Image ID:{image.getId()}',
                                                dataset=new_dataset,
                                                source_image_id=image.getId())

            omero.create_image_from_numpy_array(connection=conn,
                                                data=aip_data,
                                                image_name=f'{new_image_name}_AIP',
                                                image_description=f'Source Image ID:{image.getId()}',
                                                dataset=new_dataset,
                                                source_image_id=image.getId())

        print(f'Processed image {counter}')

finally:
    conn.close()
    print('Done')
