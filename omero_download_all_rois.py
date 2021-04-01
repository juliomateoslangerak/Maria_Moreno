import numpy as np
import omero_toolbox as omero
from getpass import getpass
from skimage import draw


# Define variables
HOST = 'omero.mri.cnrs.fr'
PORT = 4064
DOWNLOAD_DIR = '/home/julio/Downloads'


try:
    # Open the connection to OMERO
    conn = omero.open_connection(username=input("Username: "),
                                 password=getpass("OMERO Password: ", None),
                                 host=str(input('server (omero.mri.cnrs.fr): ') or HOST),
                                 port=int(input('port (4064): ') or PORT),
                                 group=input("Group: "))

    dataset_id = int(input('Dataset ID: '))

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

    counter = 0
    for image in images:
        result = roi_service.findByImage(image.getId(), None)
        for roi in result.rois:
            shape = roi.getPrimaryShape()
            try:
                shape_comment = shape.getTextValue()._val
            except AttributeError:
                shape_comment = None
            data = omero.get_shape_intensities(image, shape)

            # try:
            #     omero.create_image_from_numpy_array(conn, data, f'{image.getName()}_{shape_comment}',
            #                                         image_description=f'Source image_id:{image.getId()}',
            #                                         dataset=new_dataset)
            # except Exception as e:
            #     pass
            np.save(file=f'{DOWNLOAD_DIR}/{image.getName().npy}')

        counter += 1
        print(f'Processed image {counter}')

finally:
    conn.close()
    print('Done')
