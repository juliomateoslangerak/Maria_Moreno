import logging
import time

import numpy as np
import omero_toolbox as omero
from getpass import getpass
import subprocess
from threading import Thread

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

# Define variables
HOST = 'omero.mri.cnrs.fr'
PORT = 4064
TEMP_DIR = '/run/media/julio/DATA/Audrey/temp'
# TEMP_DIR = '/run/media/julio/DATA/Maria/temp'
ILASTIK_PATH = '/home/julio/Apps/ilastik-1.3.3post3-Linux/run_ilastik.sh'
PROJECT_PATH = '/run/media/julio/DATA/Audrey/projects/test_project_v02.ilp'
# PROJECT_PATH = '/run/media/julio/DATA/Maria/projects/test_project_v02.ilp'

ch_names = ['fibers']

def keepAlive(conn):
    global KA
    while KA:
        conn.keepAlive()
        time.sleep(60)

def run_ilastik(ilastik_path, input_path, model_path):

    cmd = [ilastik_path,
           '--headless',
           f'--project={model_path}',
           '--export_source=Probabilities',
           '--output_format=numpy',
           # '--output_filename_format={dataset_dir}/temp_Probabilities.npy',
           '--export_dtype=uint8',
           '--output_axis_order=zctyx',
           input_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE).stdout
    except subprocess.CalledProcessError as e:
        print(f'Input command: {cmd}')
        print()
        print(f'Error: {e.output}')
        print()
        print(f'Command: {e.cmd}')
        print()


if __name__ == '__main__':
    KA = True
    try:
        # Open the connection to OMERO
        conn = omero.open_connection(username=input("Username: "),
                                     password=getpass("OMERO Password: ", None),
                                     host=str(input('server (omero.mri.cnrs.fr): ') or HOST),
                                     port=int(input('port (4064): ') or PORT),
                                     group=input("Group: ") or "novoDA")

        keepAlive_thread = Thread(target=keepAlive, args=(conn,))
        keepAlive_thread.start()

        # get tagged images in dataset
        dataset_id = int(input('Dataset ID: ') or 23775)
        dataset = omero.get_dataset(conn, dataset_id)
        project = dataset.getParent()

        new_dataset_name = f'{dataset.getName()}_ilastik_output'
        new_dataset_description = f'Source DatasetId:{dataset.getId()}'
        new_dataset = omero.create_dataset(conn,
                                           name=new_dataset_name,
                                           description=new_dataset_description,
                                           parent_project=project)

        images = dataset.listChildren()

        for image in images:
            # if image.getName() not in ["PV-cre-D2-fl_B95221-M-+_+_TH-cy5_DAPI_07062023-Deblurring-01.czi [Scene #6]",
            #                            "PV-cre-D2-fl_B95224-F-cre_+_TH-cy5_DAPI_07062023-Deblurring-04.czi [Scene #7]"]:
            #     continue
            if "label image" in image.getName() or "macro image" in image.getName():
                continue

            logger.info(f'Analyzing image {image.getName()}')

            image_data = omero.get_intensities(image)

            temp_file = f'{TEMP_DIR}/{image.getId()}.npy'
            # Fishy. Our channel is 1 and we have to transpose to input, select channel 1 and then add a dimension
            np.save(temp_file, np.expand_dims(np.transpose(image_data, (2, 0, 3, 4, 1))[..., 1], 4))
            # np.save(temp_file, np.transpose(image_data, (2, 0, 3, 4, 1)))

            conn.keepAlive()

            run_ilastik(ILASTIK_PATH, temp_file, PROJECT_PATH)

            conn.keepAlive()

            output_file = f'{TEMP_DIR}/{image.getId()}_Probabilities.npy'
            prob_data = np.load(output_file)

            # Save the output back to OMERO
            omero.create_image_from_numpy_array(connection=conn,
                                                data=prob_data,
                                                image_name=f'{image.getName()}_PROB',
                                                image_description=f'Source image:{image.getId()}',
                                                dataset=new_dataset,
                                                channel_labels=ch_names + ['background'],
                                                force_whole_planes=False
                                                )


    finally:
        KA = False
        keepAlive_thread.join()
        conn.close()
        logger.info('Done')
