from getpass import getpass
import omero_toolbox as omero

# Define variables
HOST = 'omero.mri.cnrs.fr'
PORT = 4064


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
        dataset = conn.getObject('Dataset', dataset_id)
        # dataset = omero.get_dataset(conn, dataset_id)

        images = dataset.listChildren()

        deleted_count = 0

        for image in images:
            ann_ids = [ann.id for ann in image.listAnnotations()]
            deleted_count += len(ann_ids)
            if ann_ids:
                conn.deleteObjects('Annotation', ann_ids, wait=True)

        print(f'Deleted {deleted_count} annotations')

    finally:
        conn.close()
        print('Connection closed')