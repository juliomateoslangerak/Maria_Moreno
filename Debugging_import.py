import numpy as np
import omero_toolbox as ot

array = np.load('/run/media/julio/DATA/Maria/temp/Mouse_9_Replica 2_KO_05_PC_MIP.npy')
array = np.expand_dims(array,axis=0)

conn = ot.open_connection('facility_staff_1', 'facility_staff_1_pw', '192.168.56.101', 4064)

try:
    image = ot.create_image_from_numpy_array(conn, array, 'test')

finally:
    conn.close()

# __________________________
# import numpy as np
# from itertools import product
# from omero.gateway import BlitzGateway
#
# conn = BlitzGateway(username='facility_staff_1',
#                     passwd='facility_staff_1_pw',
#                     host='192.168.56.101',
#                     port=4064)
# conn.connect()
#
# image_data_1 = np.ones(shape=[1, 2, 1, 4000, 4000], dtype='float16')
#
# # create a plane generator
# zct_list = list(product(range(image_data_1.shape[0]),
#                         range(image_data_1.shape[1]),
#                         range(image_data_1.shape[2])))
# zct_generator = (image_data_1[z, c, t, :, :] for z, c, t in zct_list)
#
# try:
#     omero_image_1 = conn.createImageFromNumpySeq(zct_generator,
#                                                  imageName='Image_1',
#                                                  sizeZ=image_data_1.shape[0],
#                                                  sizeC=image_data_1.shape[1],
#                                                  sizeT=image_data_1.shape[2],
#                                                  )
#
# finally:
#     conn.close()
