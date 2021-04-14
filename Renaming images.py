import omero_toolbox as omero
from getpass import getpass

HOST = 'omero.mri.cnrs.fr'
PORT = 4064

tg_list = [174,
           362,
           363,
           369,
           371,
           601,
           703,
           800,
           ]
ig_list = [359,
           360,
           364,
           801,
           802,
           ]


if __name__ == '__main__':
    try:
        # Open the connection to OMERO
        conn = omero.open_connection(username=input("Username: "),
                                     password=getpass("OMERO Password: ", None),
                                     host=str(input('server (omero.mri.cnrs.fr): ') or HOST),
                                     port=int(input('port (4064): ') or PORT),
                                     group=input("Group: "))

        # get tagged images in dataset
        dataset_id = int(input('Dataset ID: '))
        dataset = omero.get_dataset(conn, dataset_id)

        images = dataset.listChildren()
        for image in images:
            image_name = image.getName()
            if int(image_name[6:9]) in tg_list:
                new_image_name = f'{image_name}_TG'
            elif int(image_name[6:9]) in ig_list:
                new_image_name = f'{image_name}_IG'
            else:
                raise ValueError()
            image.setName(new_image_name)
            image.save()

    finally:
        conn.close()


