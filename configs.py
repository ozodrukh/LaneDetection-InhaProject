from numpy import pi

PRODUCTION_MODE = True
LOCAL_MODE = False if PRODUCTION_MODE else True
GUI_MODE = LOCAL_MODE

resize = (300, 300)

video_source = "/Users/ozz/Documents/Projects/opencv-py/data/outcpp.avi"
camera_source = 0

camera_target = video_source if LOCAL_MODE else camera_source

if not PRODUCTION_MODE:
    single_image_source = "/Users/ozz/Desktop/Screen Shot 2019-05-15 at 5.24.50 PM.png"
    single_image_test = LOCAL_MODE and single_image_source is not None

params = {
    'paused': False,
    'fps_counter': True,
    'morph': 'gradient',
    'kernel_width': 0,
    'kernel_max_width': 0,
    'kernel_height': 1,
    'current_frame': None,
}

modified_hough_params = {
    'hough.rho': 1,
    'hough.theta': pi / 180,
    'hough.threshold': 75,
    'hough.min_distance': 64,
    'hough.max_line_gap': 60
}

params.update(modified_hough_params)
