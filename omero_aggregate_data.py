import logging

import argh
import numpy as np
import omero_toolbox as omero

import pandas as pd

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

# Define variables
TEMP_DIR = '/home/julio/temp'

NR_CHANNELS = 3

if NR_CHANNELS == 3:
    PROJECT_PATH = './models/HippocampalGliosis_v1.ilp'
    ch_names = ['Microglie', 'Astrocyte', 'Neurone']

    # Probability image is referring to channels in aip_image as follows:
    # (object_ch, prb_ch)
    object_ch_match = [(0, 0),
                       (1, 1),
                       (2, 2),
                       ]
    ch_bg_match = [(0, 3),
                   (1, 3),
                   (2, 3)
                   ]

    segmentation_thr = [180,
                        100,
                        180,
                        200]
    upper_correction_factors = [1,
                                1,
                                1,
                                1]
    lower_correction_factors = [0.8,
                                0.8,
                                0.8,
                                1]

elif NR_CHANNELS == 2:
    PROJECT_PATH = './models/Neuronal_death_v2.ilp'
    ch_names = ['Nuclei', 'Neurons_F1B']

    # Probability image is referring to channels in aip_image as follows:
    # (object_ch, prb_ch)
    object_ch_match = [(0, 0),
                       (1, 1),
                       ]
    ch_bg_match = [(0, 2),
                   (1, 2),
                   ]

    segmentation_thr = [150,
                        100,
                        200]
    upper_correction_factors = [1,
                                1,
                                1]
    lower_correction_factors = [0.8,
                                0.2,
                                1]

def run(user, password, dataset, group='Hippocampal Gliosis CD3', host='omero.mri.cnrs.fr', port=4064):
    try:
        # Open the connection to OMERO
        conn = omero.open_connection(username=user,
                                     password=password,
                                     host=host,
                                     port=int(port),
                                     group=group)

        conn.c.enableKeepAlive(60)

        # get tagged images in dataset
        dataset_id = int(dataset)
        dataset = omero.get_dataset(conn, dataset_id)
        project = dataset.getParent()

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
        table_col_values = [[] for _ in range(len(table_col_names))]

        for counter, image_root_name in enumerate(image_root_names):
            logger.info(f'Analyzing image {image_root_name}')

            mip_image = conn.getObject('Image', images_names_ids[f'{image_root_name}_MIP'])
            mip_data = omero.get_intensities(mip_image)
            aip_image = conn.getObject('Image', images_names_ids[f'{image_root_name}_AIP'])
            aip_data = omero.get_intensities(aip_image)

            # Filling data table
            name_md = image_root_name.strip()
            name_md = name_md.replace(' ', '_').split('_')

            table_col_values[0].append(aip_image)  # 'image_id'
            table_col_values[1].append(image_root_name)  # 'image_name'
            table_col_values[2].append(name_md[0])  # 'mouse_nr'
            table_col_values[3].append(name_md[1])  # 'replica_nr'
            table_col_values[4].append(name_md[2])  # 'genotype'
            table_col_values[5].append(name_md[3])  # 'treatment'

            # Some basic measurements
            roi_area = np.count_nonzero(aip_data[0, 0, 0, ...])
            table_col_values[6].append(roi_area)  # 'roi_area'

            # We were downloading the images without the z dimension, so we have to remove it here
            # mip_data = mip_data.squeeze(axis=0)

            temp_file = f'{TEMP_DIR}/temp_array.npy'
            np.save(temp_file, mip_data)

            run_ilastik(ILASTIK_PATH, temp_file, PROJECT_PATH)

            output_file = f'{TEMP_DIR}/temp_array_Probabilities.npy'
            prob_data = np.load(output_file)

            # Save the output back to OMERO
            omero.create_image_from_numpy_array(connection=conn,
                                                data=prob_data,
                                                image_name=f'{mip_image.getName()}_PROB',
                                                image_description=f'Source Image ID:{mip_image.getId()}',
                                                dataset=new_dataset,
                                                channel_labels=ch_names + ['background'],
                                                force_whole_planes=False
                                                )

            prob_data = prob_data.squeeze()
            aip_data = aip_data.squeeze()

            for object_ch, bg_ch in zip(object_ch_match, ch_bg_match):
                # Keep connection alive
                conn.keepAlive()
                # Calculate object properties on the objects
                object_labels = segment_channel(channel=prob_data[object_ch[1]], threshold=segmentation_thr[object_ch[1]])
                object_properties = compute_channel_spots_properties(channel=aip_data[object_ch[0]], label_channel=object_labels)
                object_df = pd.DataFrame(object_properties)

                # Calculate properties of the background
                bg_labels = segment_channel(channel=prob_data[bg_ch[1]], threshold=segmentation_thr[bg_ch[1]])
                bg_properties = compute_channel_spots_properties(channel=aip_data[bg_ch[0]], label_channel=bg_labels)
                bg_df = pd.DataFrame(bg_properties)

                # Save dataframes as csv attachments to the images
                object_df.to_csv(f'{TEMP_DIR}/ch{object_ch[0]}_object_df.csv')
                object_csv_ann = omero.create_annotation_file_local(
                    connection=conn,
                    file_path=f'{TEMP_DIR}/ch{object_ch[0]}_object_df.csv',
                    description=f'Data corresponding to the objects on channel {object_ch[0]}')
                omero.link_annotation(aip_image, object_csv_ann)

                bg_df.to_csv(f'{TEMP_DIR}/ch{bg_ch[0]}_bg_df.csv')
                bg_csv_ann = omero.create_annotation_file_local(
                    connection=conn,
                    file_path=f'{TEMP_DIR}/ch{bg_ch[0]}_bg_df.csv',
                    description=f'Data corresponding to the background on channel {bg_ch[0]}')
                omero.link_annotation(aip_image, bg_csv_ann)

                if len(object_df) > 0:
                    table_col_values[table_col_names.index(f'roi_intensity_{ch_names[object_ch[0]]}')].append(np.sum(aip_data[object_ch[0]]).item())
                    table_col_values[table_col_names.index(f'object_count_{ch_names[object_ch[0]]}')].append(len(object_df))

                    table_col_values[table_col_names.index(f'mean_area_{ch_names[object_ch[0]]}')].append(object_df['area'].mean().item())
                    table_col_values[table_col_names.index(f'median_area_{ch_names[object_ch[0]]}')].append(object_df['area'].median().item())
                    table_col_values[table_col_names.index(f'sum_area_{ch_names[object_ch[0]]}')].append(object_df['area'].sum().item())
                    table_col_values[table_col_names.index(f'sum_intensity_{ch_names[object_ch[0]]}')].append(object_df['integrated_intensity'].sum().item())
                    table_col_values[table_col_names.index(f'mean_intensity_{ch_names[object_ch[0]]}')].append(object_df['integrated_intensity'].sum().item() /
                                                                                                               object_df['area'].sum().item())
                    table_col_values[table_col_names.index(f'sum_area_bg_{ch_names[object_ch[0]]}')].append(bg_df['area'].sum().item())
                    table_col_values[table_col_names.index(f'sum_intensity_bg_{ch_names[object_ch[0]]}')].append(bg_df['integrated_intensity'].sum().item())
                    table_col_values[table_col_names.index(f'mean_intensity_bg_{ch_names[object_ch[0]]}')].append(bg_df['integrated_intensity'].sum().item() /
                                                                                                                  bg_df['area'].sum().item())
                else:
                    logger.warning(f'No objects were detected for image {image_root_name}')

                    table_col_values[table_col_names.index(f'roi_intensity_{ch_names[object_ch[0]]}')].append(0)
                    table_col_values[table_col_names.index(f'object_count_{ch_names[object_ch[0]]}')].append(0)

                    table_col_values[table_col_names.index(f'mean_area_{ch_names[object_ch[0]]}')].append(0)
                    table_col_values[table_col_names.index(f'median_area_{ch_names[object_ch[0]]}')].append(0)
                    table_col_values[table_col_names.index(f'sum_area_{ch_names[object_ch[0]]}')].append(0)
                    table_col_values[table_col_names.index(f'sum_intensity_{ch_names[object_ch[0]]}')].append(0)
                    table_col_values[table_col_names.index(f'mean_intensity_{ch_names[object_ch[0]]}')].append(0)
                    table_col_values[table_col_names.index(f'sum_area_bg_{ch_names[object_ch[0]]}')].append(0)
                    table_col_values[table_col_names.index(f'sum_intensity_bg_{ch_names[object_ch[0]]}')].append(0)
                    table_col_values[table_col_names.index(f'mean_intensity_bg_{ch_names[object_ch[0]]}')].append(0)

            logger.info(f'Processed image {counter}')

        table = omero.create_annotation_table(connection=conn,
                                              table_name='Aggregated_measurements',
                                              column_names=table_col_names,
                                              column_descriptions=table_col_names,
                                              values=table_col_values,
                                              )
        omero.link_annotation(dataset, table)

    finally:
        conn.close()
        logger.info('Done')


if __name__ == '__main__':
    argh.dispatch_command(run)