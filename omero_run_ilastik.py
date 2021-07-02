import logging

import argh
import numpy as np
import omero_toolbox as omero
import subprocess
from skimage.filters import threshold_otsu, apply_hysteresis_threshold
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, cube, disk

import pandas as pd

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

# Define variables
TEMP_DIR = '/home/ubuntu/temp'
ILASTIK_PATH = '/opt/ilastik-1.3.3post3-Linux/run_ilastik.sh'
# PROJECT_PATH = './models/HippocampalGliosis_v1.ilp'
PROJECT_PATH = './models/Neuronal_death_v2.ilp'

# Probability image is referring to channels in aip_image as follows:
# (object_ch, prb_ch)
# object_ch_match = [(0, 0),
                #    (1, 1),
                #    (2, 2),
                #    ]
object_ch_match = [(0, 0),
                   (1, 1),
                   ]
# ch_bg_match = [(0, 3),
            #    (1, 3),
            #    (2, 3)
            #    ]
ch_bg_match = [(0, 2),
               (1, 2),
               ]

# ch_names = ['Microglie', 'Astrocyte', 'Neurone']
ch_names = ['Nuclei', 'Neurons_F1B']

# segmentation_thr = [180,
#                     100,
#                     180,
#                     200]
segmentation_thr = [150,
                    100,
                    200]
# upper_correction_factors = [1,
#                             1,
#                             1,
#                             1]
upper_correction_factors = [1,
                            1,
                            1]
# lower_correction_factors = [0.8,
#                             0.8,
#                             0.8,
#                             1]
lower_correction_factors = [0.8,
                            0.2,
                            1]


def run_ilastik(ilastik_path, input_path, model_path):

    cmd = [ilastik_path,
           '--headless',
           f'--project={model_path}',
           '--export_source=Probabilities',
           '--output_format=numpy',
           # '--output_filename_format={dataset_dir}/{nickname}_Probabilities.npy',
           '--export_dtype=uint8',
           # '--output_axis_order=zctyx',
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


def segment_channel(channel, threshold=None, min_distance=2, remove_border=False, low_corr_factor=1, high_corr_factor=1):
    """Segment a channel (3D numpy array)
    """
    if threshold is None:
        threshold = threshold_otsu(channel)

    thresholded = apply_hysteresis_threshold(channel,
                                             low=threshold * low_corr_factor,
                                             high=threshold * high_corr_factor
                                             )

    thresholded = closing(thresholded, disk(min_distance))
    if remove_border:
        thresholded = clear_border(thresholded)
    return label(thresholded)


def segment_image(image,
                  thresholds=None,
                  low_corr_factors=None,
                  high_corr_factors=None):
    """Segment an image and return a labels object.
    Image must be provided as cyx numpy array
    """
    if len(image.shape) < 3:
        image = np.expand_dims(image, 0)

    if low_corr_factors is None:
        low_corr_factors = [.95] * image.shape[0]
    if high_corr_factors is None:
        high_corr_factors = [1.05] * image.shape[0]

    if len(high_corr_factors) != image.shape[0] or len(low_corr_factors) != image.shape[0]:
        raise Exception('The number of correction factors does not match the number of channels.')

    # We create an empty array to store the output
    labels_image = np.zeros(image.shape, dtype=np.uint16)
    for c in range(image.shape[0]):
        threshold = thresholds[c] if thresholds is not None else None
        labels_image[c, ...] = segment_channel(image[c, ...],
                                               threshold=threshold,
                                               low_corr_factor=low_corr_factors[c],
                                               high_corr_factor=high_corr_factors[c])
    return labels_image


def compute_channel_spots_properties(channel, label_channel):
    """Analyzes and extracts the properties of a single channel"""

    ch_properties = []
    logger.info(f'label_channel dims: {label_channel.shape}')
    logger.info(f'channel dims: {channel.shape}')
    regions = regionprops(label_channel, channel)

    for region in regions:
        ch_properties.append({'label': region.label,
                              'area': region.area,
                              'centroid_x': region.centroid[1],
                              'centroid_y': region.centroid[0],
                              'eccentricity': region.eccentricity,
                              'perimeter': region.perimeter,
                              'max_intensity': region.max_intensity,
                              'mean_intensity': region.mean_intensity,
                              'min_intensity': region.min_intensity,
                              'integrated_intensity': region.mean_intensity * region.area
                              })

    return ch_properties


def compute_spots_properties(image, labels):
    """Computes a number of properties for the PSF-like spots found on an image provided they are segmented"""
    # TODO: Verify dimensions of image and labels are the same
    properties = []

    for c in range(image.shape[0]):  # TODO: Deal with Time here
        pr = compute_channel_spots_properties(channel=image[c, :, :],
                                              label_channel=labels[c, :, :],
                                              )
        properties.append(pr)

    return properties


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

        new_dataset_name = f'{dataset.getName()}_ilastik_output'
        new_dataset_description = f'Source Dataset ID: {dataset.getId()}'
        new_dataset = omero.create_dataset(conn,
                                           name=new_dataset_name,
                                           description=new_dataset_description,
                                           parent_project=project)

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

            # We were downloading the images without the z dimension so we have to remove it here
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