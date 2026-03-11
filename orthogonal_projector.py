from getpass import getpass

import omero_toolbox as ot
from omero.model import Point, Line
import numpy as np
from scipy import special
from scipy.optimize import curve_fit, fsolve

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image

from skimage.transform import rescale

username = "mateos"
HOST = 'omero.mri.cnrs.fr'
PORT = 4064
GROUP = "Demos life-SR"
dataset_id = 29758
# voxel_size = (100.0, 41.5, 41.5)  # Voxel size in nm (z, y, x)
voxel_size = None
# z_size_override = 0.100  # Voxel size in nm for z-axis
z_size_override = None
# rescale_z = False  # Whether to rescale z-axis to match x and y voxel sizes
rescale_z = True
do_3D = False
# do_3D = True

from credentials import pw

def airy_fun(
    x: np.ndarray, centre: np.float64, amp: np.float64
) -> np.ndarray:  # , exp):  # , amp, bg):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            (x - centre) == 0,
            amp * 0.5**2,
            amp * (special.j1(x - centre) / (x - centre)) ** 2,
        )


def gaussian_fun(x, background, amplitude, center, sd):
    gauss = np.exp(-np.power(x - center, 2.0) / (2 * np.power(sd, 2.0)))
    return background + (amplitude - background) * gauss


def fit_gaussian(profile, guess=None):
    if guess is None:
        guess = [profile.min(), profile.max(), profile.argmax(), 0.8]
    x = np.linspace(0, profile.shape[0], profile.shape[0], endpoint=False)
    popt, pcov, infodict, mesg, ier = curve_fit(
        f=gaussian_fun, xdata=x, ydata=profile, p0=guess, maxfev=5000, full_output=True
    )

    if ier not in [1, 2, 3, 4]:
        print(f"No gaussian fit found. Reason: {mesg}")

    fitted_profile = gaussian_fun(x, popt[0], popt[1], popt[2], popt[3])
    fwhm = popt[3] * 2.35482

    # Calculate the fit quality using the coefficient of determination (R^2)
    y_mean = np.mean(profile)
    sst = np.sum((profile - y_mean) ** 2)
    ssr = np.sum((profile - fitted_profile) ** 2)
    cd_r2 = 1 - (ssr / sst)

    return fitted_profile, cd_r2, fwhm, popt


def fit_airy(profile, guess=None):
    profile = (profile - profile.min()) / (profile.max() - profile.min())
    if guess is None:
        guess = [profile.argmax(), 4 * profile.max()]
    x = np.linspace(0, profile.shape[0], profile.shape[0], endpoint=False)
    popt, pcov, infodict, mesg, ier = curve_fit(
        f=airy_fun, xdata=x, ydata=profile, p0=guess, full_output=True
    )

    if ier not in [1, 2, 3, 4]:
        print(f"No airy fit found. Reason: {mesg}")

    fitted_profile = airy_fun(x, popt[0], popt[1])

    # Ensure the profile is normalized
    fitted_profile = (fitted_profile - fitted_profile.min()) / (fitted_profile.max() - fitted_profile.min())

    # Calculate the FWHM
    def _f(d):
        return airy_fun(d, popt[0], popt[1]) - 0.5  # Half maximum is 0.5 for normalized profile

    guess = np.array([fitted_profile.argmax() - 1, fitted_profile.argmax() + 1])
    v = fsolve(_f, guess)
    fwhm = abs(v[1] - v[0])

    # Calculate the fit quality using the coefficient of determination (R^2)
    y_mean = np.mean(profile)
    sst = np.sum((profile - y_mean) ** 2)
    ssr = np.sum((profile - fitted_profile) ** 2)
    cd_r2 = 1 - (ssr / sst)

    return fitted_profile, cd_r2, fwhm, popt

def calculate_fwhm_raw(profile):
    # Normalize the profile
    profile = (profile - profile.min()) / (profile.max() - profile.min())

    # Find the half maximum value
    half_max = 0.5

    # Find indices where the profile crosses the half maximum
    indices = np.where(profile >= half_max)[0]

    if len(indices) < 2:
        print("Unable to calculate FWHM: insufficient crossings.")
        return np.nan

    return indices[-1] - indices[0]


def extract_profiles(
        image: np.ndarray,
        coords: tuple,
        l: int,
        do_3D: bool,
        voxel_size=(1.0, 1.0, 1.0),
        normalize_intensities: bool = True
) -> dict:
    """
    Extracts 1D intensity profiles along each axis from a 3D image
    centered at the given coordinates and returns plots as RGB arrays.

    Parameters
    ----------
    image : np.ndarray
        3D numpy array (z, y, x) of intensity values
    coords : tuple of int
        (z, y, x) coordinates of the center point
    l : int
        Length of profiles (must be odd for symmetry)
    do_3D : bool
        Whether to extract 3D profiles
    normalize_intensities : bool
        Whether to normalize intensities to [0, 1] range
    voxel_size : tuple of float
        (sz, sy, sx) spacing factors for each axis (e.g. voxel size)

    Returns
    -------
    profiles : dict
        Dictionary with keys "z", "y", "x" and values as 2048x2048x4 uint8 arrays
    """
    zc, yc, xc = coords
    half = l // 2

    # Clip ranges so they don’t go out of bounds
    if do_3D:
        zmin, zmax = max(0, zc-half), min(image.shape[0], zc+half+1)
    ymin, ymax = max(0, yc-half), min(image.shape[1], yc+half+1)
    xmin, xmax = max(0, xc-half), min(image.shape[2], xc+half+1)

    # Extract profiles
    if do_3D:
        profile_z = image[zmin:zmax, yc, xc]
    profile_y = image[zc, ymin:ymax, xc]
    profile_x = image[zc, yc, xmin:xmax]

    def normalize(arr):
        if arr.max() > arr.min():
            return (arr - arr.min()) / (arr.max() - arr.min())
        else:
            return np.zeros_like(arr)

    if normalize_intensities:
        if do_3D:
            profile_z = normalize(profile_z)
        profile_y = normalize(profile_y)
        profile_x = normalize(profile_x)

    # Prepare x-axis for each profile
    if do_3D:
        xz = np.arange(zmin - zc, zmax - zc) * voxel_size[0]
    xy = np.arange(ymin - yc, ymax - yc) * voxel_size[1]
    xx = np.arange(xmin - xc, xmax - xc) * voxel_size[2]

    # Fit profiles
    if do_3D:
        fit_z, cd_r2_z, fwhm_z, popt_z = fit_gaussian(profile_z)
    fit_y, cd_r2_y, fwhm_y, popt_y = fit_gaussian(profile_y)
    fit_x, cd_r2_x, fwhm_x, popt_x = fit_gaussian(profile_x)

    # correct the FWHM values to match the voxel size
    if do_3D:
        fwhm_z = fwhm_z * voxel_size[0]
    fwhm_y = fwhm_y * voxel_size[1]
    fwhm_x = fwhm_x * voxel_size[2]

    # get raw FWHM values
    if do_3D:
        fwhm_z_raw = calculate_fwhm_raw(profile_z) * voxel_size[0]
    fwhm_y_raw = calculate_fwhm_raw(profile_y) * voxel_size[1]
    fwhm_x_raw = calculate_fwhm_raw(profile_x) * voxel_size[2]

    def plot_profile(x, y, fit_y, title, color, fit_color, labels=None):
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.plot(x, y, color=color)
        ax.plot(x, fit_y, color=fit_color, linestyle='--', linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Offset")
        ax.set_ylabel("Normalized Int")

        # Add labels if provided
        if labels:
            text_lines = [f"{k}: {v:.3f}" if isinstance(v, (float, int)) else f"{k}: {v}"
                          for k, v in labels.items()]
            text = "\n".join(text_lines)
            ax.text(0.95, 0.95, text,
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

        fig.tight_layout()

        # Render figure to RGB array
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
        img = img.reshape(canvas.get_width_height()[::-1] + (4,))
        # Resize to 400x400
        img = np.array(Image.fromarray(img))

        plt.close(fig)

        return img

    profiles = {}
    if do_3D:
        profiles["z"] = plot_profile(
            xz, profile_z, fit_z, "Z profile", "r", "g", {
                "FWHM_raw": f"{fwhm_z_raw:.2f}",
                "FWHM_fit": f"{fwhm_z:.2f}",
                "R^2": f"{cd_r2_z:.2f}",
            }
        )
    profiles["y"] = plot_profile(
            xy, profile_y, fit_y, "Y profile", "r", "g", {
                "FWHM_raw": f"{fwhm_y_raw:.2f}",
                "FWHM_fit": f"{fwhm_y:.2f}",
                "R^2": f"{cd_r2_y:.3f}",
            }
        )
    profiles["x"] = plot_profile(
            xx, profile_x, fit_x, "X profile", "r", "g", {
                "FWHM_raw": f"{fwhm_x_raw:.2f}",
                "FWHM_fit": f"{fwhm_x:.2f}",
                "R^2": f"{cd_r2_x:.3f}",
            }
        )

    return profiles


try:
    # Open the connection to OMERO
    conn = ot.open_connection(
        username=str(input(f"Username ({username}): ") or username),
        password=str(input("OMERO Password: ") or pw),
        host=str(input("server (omero.mri.cnrs.fr): ") or HOST),
        port=int(input("port (4064): ") or PORT),
        group=str(input(f"Group ({GROUP}): ") or GROUP),
    )

    dataset_id = int(input("Dataset ID: ") or dataset_id)
    # dataset_id = dataset_id

    dataset = ot.get_dataset(conn, dataset_id)
    images = dataset.listChildren()

    roi_service = conn.getRoiService()

    for image in images:
        image_intensities = None

        rois = roi_service.findByImage(image.getId(), None).rois
        for roi in rois:
            shape = roi.getPrimaryShape()
            if isinstance(shape, Point):
                try:
                    z_pos = shape.getTheZ()._val
                except AttributeError:
                    z_pos = None
                t_pos = 0  # Assuming time point is always 0, can be modified if needed
                try:
                    c_pos = shape.getTheC()._val
                except AttributeError:
                    c_pos = None
                y_pos = shape.getY()._val
                x_pos = shape.getX()._val
                try:
                    shape_comment = shape.getTextValue()._val
                except AttributeError:
                    shape_comment = f"Point_at_{z_pos}_{y_pos:.2f}_{x_pos:.2f}"

                if image_intensities is None:
                    image_intensities = ot.get_intensities(image)
                    # If we find a time lapse, this is probably the phases in the raw image, and we want to
                    # average them
                    if image_intensities.shape[2] > 1:
                        image_intensities = np.mean(image_intensities, axis=2, keepdims=True)
                    if image_intensities.min() < 0:
                        image_intensities = np.clip(image_intensities, 0, None)
                    if voxel_size is None:
                        voxel_size = ot.get_pixel_size(image, order='zyx')
                    if z_size_override is not None:
                        voxel_size = (z_size_override, voxel_size[1], voxel_size[2])

                if c_pos is None:
                    c_positions = range(image_intensities.shape[1])
                elif isinstance(c_pos, int):
                    c_positions = [c_pos]
                else:
                    raise ValueError("c_pos should be an int or a list of ints")

                if z_pos is None:
                    z_positions = range(image_intensities.shape[0])
                elif isinstance(z_pos, int):
                    z_positions = [z_pos]
                else:
                    raise ValueError("z_pos should be an int or a list of ints")

                for z_pos in z_positions:
                    for c_pos in c_positions:
                        # dims in order z, c, t, y, x
                        yx_slice = image_intensities[z_pos, c_pos, t_pos, :, :]
                        yx_slice = np.reshape(yx_slice, (1, 1, 1, yx_slice.shape[0], yx_slice.shape[1]))
                        yx_image = ot.create_image_from_numpy_array(
                            connection=conn,
                            data=yx_slice,
                            image_name=f"{image.getName()}_{shape_comment}_C{c_pos}_Z{z_pos}_yx",
                            image_description=f"yx Orthogonal slice at Z={z_pos}, C={c_pos}, T={t_pos}. source imageid:{image.getId()}",
                            # channel_labels=None,
                            dataset=dataset,
                            source_image_id=image.getId(),
                            # channels_list=None,
                            force_whole_planes=True
                        )
                        yx_image.unload()
                        yx_line_roi = ot.create_roi(
                            connection=conn,
                            image=yx_image,
                            shapes=[
                                ot.create_shape_line(
                                    x1_pos=0,
                                    y1_pos=y_pos,
                                    x2_pos=yx_slice.shape[4],
                                    y2_pos=y_pos,
                                    stroke_color=(255, 0, 0, 200),
                                    stroke_width=0.5,
                                ),
                                ot.create_shape_line(
                                    x1_pos=x_pos,
                                    y1_pos=0,
                                    x2_pos=x_pos,
                                    y2_pos=yx_slice.shape[3],
                                    stroke_color=(255, 0, 0, 200),
                                    stroke_width=0.5,
                                )
                            ]
                        )

                        zx_slice = image_intensities[:, c_pos, t_pos, int(y_pos), :]
                        if rescale_z:
                            zx_slice = rescale(
                                zx_slice,
                                (voxel_size[0] / voxel_size[2], 1),
                                anti_aliasing=True,
                                preserve_range=True
                            )
                        zx_slice = zx_slice.astype(image_intensities.dtype)
                        zx_slice = np.reshape(zx_slice, (1, 1, 1, zx_slice.shape[0], zx_slice.shape[1]))
                        zx_image = ot.create_image_from_numpy_array(
                            connection=conn,
                            data=zx_slice,
                            image_name=f"{image.getName()}_{shape_comment}_C{c_pos}_Z{z_pos}_zx",
                            image_description=f"zx Orthogonal slice at Z={z_pos}, C={c_pos}, T={t_pos}. source image:{image.getId()}",
                            # channel_labels=None,
                            dataset=dataset,
                            source_image_id=image.getId(),
                            # channels_list=None,
                            force_whole_planes=True
                        )
                        zx_image.unload()
                        zx_line_roi = ot.create_roi(
                            connection=conn,
                            image=zx_image,
                            shapes=[
                                ot.create_shape_line(
                                    x1_pos=0,
                                    y1_pos=(z_pos - 0.5) * (voxel_size[0] / voxel_size[2]),
                                    x2_pos=zx_slice.shape[4],
                                    y2_pos=(z_pos - 0.5) * (voxel_size[0] / voxel_size[2]),
                                    stroke_color=(255, 0, 0, 200),
                                    stroke_width=0.5,
                                ),
                                ot.create_shape_line(
                                    x1_pos=x_pos,
                                    y1_pos=0,
                                    x2_pos=x_pos,
                                    y2_pos=zx_slice.shape[3],
                                    stroke_color=(255, 0, 0, 200),
                                    stroke_width=0.5,
                                )
                            ]
                        )

                        zy_slice = image_intensities[:, c_pos, t_pos, :, int(x_pos)]
                        if rescale_z:
                            zy_slice = rescale(
                                zy_slice,
                                (voxel_size[0] / voxel_size[1], 1),
                                anti_aliasing=True,
                                preserve_range=True
                            )
                        zy_slice = zy_slice.astype(image_intensities.dtype)
                        zy_slice = np.transpose(zy_slice, (1, 0))  # Transpose to get the correct orientation
                        zy_slice = np.reshape(zy_slice, (1, 1, 1, zy_slice.shape[0], zy_slice.shape[1]))
                        zy_image = ot.create_image_from_numpy_array(
                            connection=conn,
                            data=zy_slice,
                            image_name=f"{image.getName()}_{shape_comment}_C{c_pos}_Z{z_pos}_zy",
                            image_description=f"zy Orthogonal slice at Z={z_pos}, C={c_pos}, T={t_pos}. source imageid:{image.getId()}",
                            # channel_labels=None,
                            dataset=dataset,
                            source_image_id=image.getId(),
                            # channels_list=None,
                            force_whole_planes=True
                        )
                        zy_image.unload()
                        zy_line_roi = ot.create_roi(
                            connection=conn,
                            image=zy_image,
                            shapes=[
                                ot.create_shape_line(
                                    x1_pos=0,
                                    y1_pos=y_pos,
                                    x2_pos=zy_slice.shape[4],
                                    y2_pos=y_pos,
                                    stroke_color=(255, 0, 0, 200),
                                    stroke_width=0.5,
                                ),
                                ot.create_shape_line(
                                    x1_pos=(z_pos - 0.5) * (voxel_size[0] / voxel_size[1]),
                                    y1_pos=0,
                                    x2_pos=(z_pos - 0.5) * (voxel_size[0] / voxel_size[1]),
                                    y2_pos=zy_slice.shape[3],
                                    stroke_color=(255, 0, 0, 200),
                                    stroke_width=0.5,
                                )
                            ]
                        )

                        profiles = extract_profiles(
                            image_intensities[:, c_pos, t_pos, :, :],
                            (z_pos, int(y_pos), int(x_pos)),
                            21,
                            voxel_size=voxel_size,
                            do_3D=do_3D,
                        )
                        for axis, profile_img in profiles.items():
                            # Convert to RGB for OMERO
                            omero_rgb_profile = np.transpose(profile_img, (2, 0, 1))
                            omero_rgb_profile = omero_rgb_profile.reshape(
                                (1, 4, 1, omero_rgb_profile.shape[1], omero_rgb_profile.shape[2])
                            )
                            profile_image = ot.create_image_from_numpy_array(
                                connection=conn,
                                data=omero_rgb_profile,
                                image_name=f"{image.getName()}_{shape_comment}_C{c_pos}_Z{z_pos}_{axis}_profile",
                                image_description=f"Profile along {axis} axis at Z={z_pos}, C={c_pos}, T={t_pos}. source imageid:{image.getId()}",
                                dataset=dataset,
                            )
                            profile_image.unload()

            elif isinstance(shape, Line):
                try:
                    z_pos = shape.getTheZ()._val
                except AttributeError:
                    z_pos = None
                t_pos = 0  # Assuming time point is always 0, can be modified if needed
                try:
                    c_pos = shape.getTheC()._val
                except AttributeError:
                    c_pos = None
                y_pos_1 = shape.getTheY1()._val
                y_pos_2 = shape.getTheY2()._val
                x_pos_1 = shape.getTheX1()._val
                x_pos_2 = shape.getTheX2()._val
                try:
                    shape_comment = shape.getTextValue()._val
                except AttributeError:
                    shape_comment = f"Line_at_{z_pos}_a-{y_pos_1:.2f}:{x_pos_1:.2f}_b-{y_pos_2:.2f}:{x_pos_2:.2f}"
                if image_intensities is None:
                    image_intensities = ot.get_intensities(image)
                    # If we find a time lapse, this is probably the phases in the raw image, and we want to
                    # average them
                    if image_intensities.shape[2] > 1:
                        image_intensities = np.mean(image_intensities, axis=2, keepdims=True)
                    if image_intensities.min() < 0:
                        image_intensities = np.clip(image_intensities, 0, None)
                    if voxel_size is None:
                        voxel_size = ot.get_pixel_size(image, order='zyx')
                    if z_size_override is not None:
                        voxel_size = (z_size_override, voxel_size[1], voxel_size[2])



finally:
    conn.close()
    print('Done')
