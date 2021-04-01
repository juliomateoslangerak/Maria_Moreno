import numpy as np
import omero_toolbox as omero
from getpass import getpass
from skimage import draw


# Define variables
HOST = 'omero.mri.cnrs.fr'
PORT = 4064


# Helper functions
def get_tagged_images(dataset, tag):
    images = dataset.listChildren()
    tagged_images = list()
    for image in images:
        for ann in image.listAnnotations():
            if ann.OMERO_TYPE == omero.model.TagAnnotationI and ann.getTextValue() == tag:
                tagged_images.append(image)
                break

    return tagged_images


try:
    # Open the connection to OMERO
    conn = omero.open_connection(username=input("Username: "),
                                 password=getpass("OMERO Password: ", None),
                                 host=str(input('server (omero.mri.cnrs.fr): ') or HOST),
                                 port=int(input('port (4064): ') or PORT),
                                 group=input("Group: "))

    # get tagged images in dataset
    output_dir = input('Output directory: ')
    dataset_id = input('Dataset ID: ')
    tag_text = input('Tag_text (leave empty for no filtering): ')

    dataset = conn.getObject('Dataset', dataset_id)

    if tag_text == '':
        images = list(dataset.listChildren())
    else:
        images = get_tagged_images(dataset, tag_text)

    counter = 0
    for image in images:
        data = omero.get_intensities(image)
        file_name = f'{output_dir}/{image.getName()}.npy'
        np.save(file_name, data)
        counter += 1
        print(f'Processed image {counter} of {len(images)}')
        
finally:
    conn.close()
    print('Done')
