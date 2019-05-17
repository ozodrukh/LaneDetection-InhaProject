from numpy import pi

LOCAL_MODE = True
GUI_MODE = True

resize = (300, 300)

video_source = "/Users/ozz/Documents/Projects/opencv-py/data/outcpp.avi"
camera_source = 0

camera_target = video_source

single_image_test = True
single_image_source = "/Users/ozz/Desktop/Screen Shot 2019-05-15 at 4.32.18 PM.png"

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
