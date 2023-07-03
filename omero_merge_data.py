import logging
import argh

import omero_toolbox as omero
from omero.gateway import FileAnnotationWrapper
from getpass import getpass
import os
import pandas as pd
import numpy as np

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

# Define variables
HOST = 'omero.mri.cnrs.fr'
PORT = 4064
TEMP_DIR = '/home/julio/temp'

# Probability image is referring to channels in aip_image as follows:
# (object_ch, prb_ch)
object_ch_match = [(0, 0),
                   (1, 1),
                   (2, 2),
                   ]
ch_bg_match = [(0, 3),
               (1, 3),
               (2, 3)]
# ch_bg_match = [(0, 2),
            #    (1, 2),
            #    ]
ch_names = ['Microglie', 'Astrocyte', 'Neurone']
# ch_names = ['Nuclei', 'Neurons_F1B']


def run(user, password, dataset, group='Hippocampal Gliosis CD3', host='omero.mri.cnrs.fr', port=4064):
    try:
        # Open the connection to OMERO
        conn = omero.open_connection(username=user,
                                     password=password,
                                     host=host,
                                     port=port,
                                     group=group)
        conn.c.enableKeepAlive(60)

        # get tagged images in dataset
        dataset_id = int(dataset)
        dataset = omero.get_dataset(conn, dataset_id)

        images = dataset.listChildren()
        images_names_ids = {i.getName(): i.getId() for i in images}
        image_root_names = list(set([n[:-4] for n in images_names_ids.keys()]))

        table_col_names = ['image_id',
                           'image_name',
                           'mouse_nr',
                           'replica_nr',
                           'genotype',
                           'treatment',
                           'roi_area']

        for ch_name in ch_names:
            table_col_names.extend([f'roi_intensity_{ch_name}',
                                    f'object_count_{ch_name}',
                                    f'mean_area_{ch_name}',
                                    f'median_area_{ch_name}',
                                    f'sum_area_{ch_name}',
                                    f'sum_intensity_{ch_name}',
                                    f'mean_intensity_{ch_name}',
                                    f'sum_area_bg_{ch_name}',
                                    f'sum_intensity_bg_{ch_name}',
                                    f'mean_intensity_bg_{ch_name}'
                                    ])

        table = pd.DataFrame(columns=table_col_names)

        for image_root_name in image_root_names:
            logger.info(f'Analyzing image {image_root_name}')

            image = conn.getObject('Image', images_names_ids[f'{image_root_name}_AIP'])
            aip_data = omero.get_intensities(image)
            # aip_data = np.load(str(os.path.join(TEMP_DIR, str(dataset_id), f'{image_root_name}.npy')))
            aip_data = np.squeeze(aip_data)

            # Filling data table
            name_md = image_root_name.strip()
            name_md = name_md.replace(' ', '_').split('_')

            im_table = pd.DataFrame(columns=table_col_names)

            im_table['image_id'] = [image.getId()]
            im_table['image_name'] = [image_root_name]
            im_table['mouse_nr'] = [name_md[0]]
            im_table['replica_nr'] = [name_md[1]]
            im_table['genotype'] = [name_md[2]]
            im_table['treatment'] = [name_md[3]]

            # Some basic measurements
            roi_area = np.count_nonzero(aip_data[0])
            im_table['roi_area'] = [roi_area]

            object_dfs = [None for _ in range(len(ch_names))]
            bg_dfs = [None for _ in range(len(ch_names))]

            anns = image.listAnnotations()

            for ann in anns:
                if isinstance(ann, FileAnnotationWrapper):
                    file_name = ann.getFileName()

                    local_file_path = os.path.join(TEMP_DIR, f'{image.getId()}_{image_root_name}_{file_name}')
                    with open(str(local_file_path), 'wb') as f:
                        for chunk in ann.getFileInChunks():
                            f.write(chunk)

                    if 'object_df' in file_name:
                        object_dfs[int(file_name[2])] = pd.read_csv(local_file_path)

                    elif 'bg_df' in file_name:
                        bg_dfs[int(file_name[2])] = pd.read_csv(local_file_path)

            for ch, (object_df, bg_df) in enumerate(zip(object_dfs, bg_dfs)):

                im_table[f'roi_intensity_{ch_names[ch]}'] = np.sum(aip_data[ch])

                if len(object_df) > 0:
                    im_table[f'object_count_{ch_names[ch]}'] = len(object_df)
                    im_table[f'mean_area_{ch_names[ch]}'] = object_df['area'].mean()
                    im_table[f'median_area_{ch_names[ch]}'] = object_df['area'].median()
                    im_table[f'sum_area_{ch_names[ch]}'] = object_df['area'].sum()
                    im_table[f'sum_intensity_{ch_names[ch]}'] = object_df['integrated_intensity'].sum()
                    im_table[f'mean_intensity_{ch_names[ch]}'] = object_df['integrated_intensity'].sum() / \
                                                                 object_df['area'].sum()
                else:
                    im_table[f'object_count_{ch_names[ch]}'] = 0
                    im_table[f'mean_area_{ch_names[ch]}'] = 0
                    im_table[f'median_area_{ch_names[ch]}'] = 0
                    im_table[f'sum_area_{ch_names[ch]}'] = 0
                    im_table[f'sum_intensity_{ch_names[ch]}'] = 0
                    im_table[f'mean_intensity_{ch_names[ch]}'] = 0

                    logger.warning(f'No {ch_names[ch]} were detected for image {image_root_name}')

                im_table[f'sum_area_bg_{ch_names[ch]}'] = bg_df['area'].sum()
                im_table[f'sum_intensity_bg_{ch_names[ch]}'] = bg_df['integrated_intensity'].sum()
                im_table[f'mean_intensity_bg_{ch_names[ch]}'] = bg_df['integrated_intensity'].sum() / \
                                                                bg_df['area'].sum()

            table = table.append(im_table)

        table.to_csv(os.path.join(TEMP_DIR, f'Table_output_{dataset_id}.csv'))

    finally:
        conn.close()
        logger.info('Done')


if __name__ == '__main__':
    argh.dispatch_command(run)