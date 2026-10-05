import cv2
import numpy as np
import random
import os
import json
import Shapes as sh
import time
import shutil
import argparse
import colorama
import seaborn as sns
import matplotlib.pyplot as plt
import resize_backgrounds as rb
import generate_letters as genl

DEBUG = False 
SIZE = 320
BLUR_FACTOR = [0, 4]
NOISE_FACTOR = 50 
N = 100 #images per shape
DISPLAY = False 
FREQ = 0 #frequency for display
SAVE_BOUNDS = False 
BRIGHTNESS_DIFFERENCE = 100
CONTRAST = 20
BRIGHTNESS = [0.85, 1.2]
SHAPE_SCALE = [0.2, 0.6]
SHOW_DISTRIBUTION = False

IMAGE_SIZE = [640, 640, 3]
ADD_LETTER = True 
ADD_LIGHT_GRADIENT = False 
LIGHT_PROB = 0.5
OPACITY = [0.0, 0.2]
BLEND_MODE = 'o'
LETTER_PROB = 0.5
LETTER_DATASET = False 
ONLY_LETTER = False

# NOISE_CACHE = {
#     i: np.random.normal(loc=0, scale=i, size=IMAGE_SIZE) for i in range(NOISE_FACTOR[0], NOISE_FACTOR[1] + 1)
# }
KERNEL_CACHE = {
    max(1, i): cv2.getGaussianKernel(max(1, i), 0) for i in range(BLUR_FACTOR[0], BLUR_FACTOR[1] + 1)
}

colorama.init()

SHAPES = ['Quartercircle','Semicircle','Cross','Star','Triangle','Octagon','Heptagon','Hexagon','Square',
          'Pentagon','Circle','Rectangle','Trapezoid']

SHAPES = ['Quartercircle','Semicircle','Cross','Star','Triangle', 
          'Pentagon','Circle','Rectangle']

LETTER_CLASS = {char: idx for idx, char in enumerate([str(i) for i in range(10)] + [chr(i) for i in range(ord('A'), ord('Z') + 1)])}


SCALE_OFFSET = {
    'Quartercircle': [0.1, 0.21],
    'Semicircle' : [0.0, 0], 
    'Cross': [0, 0],
    'Star': [0, 0.11],
    'Triangle': [0, 0.1], 
    'Pentagon':[0, 0],
    'Circle': [0, -0.01],
    'Rectangle': [0.1, 0]
}


Shapes_dir = {}
curdir = os.path.abspath(os.path.dirname(__file__))

Background = os.path.join(curdir, "..", "Backgrounds", "Backgrounds")
Letters = os.path.join(curdir, 'res', 'letters')
Output = os.path.join(curdir,'output', 'images')
LetterOutput = os.path.join(curdir, 'output', 'letter_images')
Labels = os.path.join(curdir, 'output', 'lables')
LetterLabels = os.path.join(curdir, 'output', 'letter_lables')
Bounds = os.path.join(curdir, 'output', 'bounds' )
LetterBounds = os.path.join(curdir, 'output', 'letter_bounds' )
Light = os.path.join(curdir, 'res', 'lights')
LetterDir = {}
# Background = os.path.abspath(r'D:\dataset\Backgrounds\Backgrounds')

Vertex_Data = json.load(open(os.path.join(curdir,'res', 'vertex.json')))

     
def progress_bar(current, total):
    total_progress = 50
    current_progress = int(50 * (current/total)) + 1

    output = 'Images: {} \n[\033[92m'.format(current) + ('').join(['=' for _ in range(current_progress)])
    output += ('').join([' ' for _ in range(total_progress - current_progress)]) + '\033[0m]'

    return output

def add_noise(image, noise_factor):
    # noise = np.random.normal(loc=0, scale=noise_factor, size=image.shape)
    # noise = NOISE_CACHE[noise_factor]
    # noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    # return noisy_image
    if noise_factor == 0:
        return image
    image = cv2.convertScaleAbs(image + np.random.randint(-noise_factor, noise_factor, image.shape))
    alpha = random.uniform(0.85, 1.2)
    beta = random.randint(-20, 20)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta= beta)
    return image

def apply_blur(image, blur_factor):
    blur_factor = max(1, blur_factor)
    kernel = KERNEL_CACHE[blur_factor]
    blur_image = cv2.filter2D(image, -1, kernel)
    blur_image = cv2.filter2D(blur_image, -1, kernel.T)
    return blur_image

def differentiate(color_a, color_b, brightness_diff):
    hue_diff = int(0.3 * brightness_diff)
    color_a[0], color_b[0] = apply_difference(color_a[0], color_b[0], hue_diff)
    color_a[2], color_b[2] = apply_difference(color_a[2], color_b[2], brightness_diff)
   
    return (color_a, color_b) if random.randint(0, 1) else (color_b, color_a)


def apply_difference(attrib_a, attrib_b, attrib_diff):
    low, high = min(attrib_a, attrib_b), max(attrib_a, attrib_b)
    if abs(high - low) > attrib_diff:
        return attrib_a, attrib_b
    
    mean = (low+ high) // 2
    if mean + attrib_diff//2 > 255: 
        mean -= (mean + attrib_diff//2) - 255

    if mean - attrib_diff//2 < 0: 
        mean += -(mean - attrib_diff//2)
    
    new_low = mean - attrib_diff//2
    new_low += 255 if new_low < 0 else 0
    new_high = mean + attrib_diff//2
    new_high -= 255 if new_high > 255 else 0
    # print(bright)
    # print(dark)
    return (new_low, new_high) if attrib_a == low else (new_high, new_low)

def draw_shape(image, shape, letter, s_scale, l_scale, translation, s_rot_angle, l_rot_angle, s_color, l_color, l_dice,  add_letter = True):
    dimension = [int(image.shape[0] * s_scale), int(image.shape[1] * s_scale)]
    try:  
        if shape == "Rectangle":
            # s_scale *= 0.4
            vertices, offset= sh.Rectangle(s_scale)if shape not in Vertex_Data else (Vertex_Data[shape], Vertex_Data[shape + '_off'])
            image, data, centroid= sh.Polygon(image, vertices, dimension, s_color, s_rot_angle, translation)
            letter_data = {'x': 0, 'y': 0, 'w': 0, 'h': 0}
            if add_letter and l_dice < LETTER_PROB: image , letter_data= sh.add_letter(image, letter, l_color, (s_scale * l_scale * offset) / 1.2, centroid, l_rot_angle)
            return image, data, letter_data

        elif shape == "Quartercircle":
            image, data, centroid = sh.Circle(image, [0 + s_rot_angle, 90 + s_rot_angle], s_scale, s_color, translation)
            l_scale = (s_scale * l_scale)/ 3.2

            letter_data = {'x': 0, 'y': 0, 'w': 0, 'h': 0}
            if add_letter and l_dice < LETTER_PROB: image , letter_data= sh.add_letter(image, letter, l_color, l_scale, centroid, l_rot_angle)
            return image, data, letter_data

        elif shape == "Semicircle":
            image, data, centroid = sh.Circle(image, [0+ s_rot_angle, 180 + s_rot_angle], s_scale, s_color, translation)
            l_scale = (s_scale * l_scale) / 2.1

            letter_data = {'x': 0, 'y': 0, 'w': 0, 'h': 0}
            if add_letter and l_dice < LETTER_PROB: image, letter_data = sh.add_letter(image, letter, l_color, l_scale, centroid, l_rot_angle)
            return image, data, letter_data

        elif shape == "Circle":
            image, data, centroid = sh.Circle(image, [0, 360], s_scale, s_color, translation)
            s_scale *= 0.75
            l_scale = (s_scale * l_scale) / 1.1

            letter_data = {'x': 0, 'y': 0, 'w': 0, 'h': 0}
            if add_letter and l_dice < LETTER_PROB: image, letter_data = sh.add_letter(image, letter, l_color, l_scale, centroid, l_rot_angle)
            return image, data, letter_data

        elif shape == "Trapezoid":
            # s_scale *= 0.4
            vertices = sh.Trapezoid() if shape not in Vertex_Data else Vertex_Data[shape]
            image , data, centroid = sh.Polygon(image, vertices, dimension, s_color, s_rot_angle, translation)
            l_scale = (s_scale * l_scale) / 1.5

            letter_data = {'x': 0, 'y': 0, 'w': 0, 'h': 0}
            if add_letter and l_dice < LETTER_PROB: image, letter_data = sh.add_letter(image, letter, l_color, l_scale, centroid, l_rot_angle)
            return image, data, letter_data

        elif shape == "Cross":
            vertices, offset = sh.Cross() if shape not in Vertex_Data else (Vertex_Data[shape], Vertex_Data[shape + '_off'])
            image , data, centeroid = sh.Polygon(image, vertices, dimension, s_color, s_rot_angle, translation)

            letter_data = {'x': 0, 'y': 0, 'w': 0, 'h': 0}
            if add_letter and l_dice < LETTER_PROB: image, letter_data = sh.add_letter(image, letter, l_color, offset * l_scale * s_scale * 1.1, centeroid, l_rot_angle)
            return image, data, letter_data

        elif shape in Vertex_Data:
            # s_scale = (s_scale* 0.4) if shape == 'Square' else s_scale 

            image, data, centroid= sh.Polygon(image, Vertex_Data[str(shape)], dimension, s_color, s_rot_angle, translation)

            modified_l_scale = (s_scale * l_scale)/ 1.8
            modified_l_scale = (s_scale * l_scale)/ 1.8 if shape == 'Square' else modified_l_scale
            modified_l_scale = (s_scale * l_scale) / 2.2 if shape == 'Triangle' else modified_l_scale
            modified_l_scale = (s_scale * l_scale)/ 2.6 if shape == "Star" else modified_l_scale

            letter_data = {'x': 0, 'y': 0, 'w': 0, 'h': 0}
            if add_letter and l_dice < LETTER_PROB: image , letter_data = sh.add_letter(image, letter, l_color, modified_l_scale, centroid, l_rot_angle)
            return image, data, letter_data
        
        else:
            print(str(shape), " does not exists")
            return image, {}

    except Exception as e: 
        if DEBUG: 
            print(e)
            exit(-1)
        return None, None, None
        
    except KeyboardInterrupt:
        print('Exiting....')
        raise 


def calculate_poly_dimension(size , shape, s_scale, rot_angle):
    dimension = [int(size[0] * s_scale), int(size[1] * s_scale)]
    
    if shape == "Rectangle":
        vertices , offset = sh.Rectangle(s_scale)
        Vertex_Data['Rectangle'] = vertices
        Vertex_Data['Rectangle_off'] = offset
        _, _ , dimension = sh.calculate_rotated_vertex(vertices, dimension, rot_angle)  
        return dimension 

    elif shape == "Trapezoid":
        vertices = sh.Trapezoid()
        Vertex_Data['Trapezoid'] = vertices
        _, _ , dimension = sh.calculate_rotated_vertex(vertices, dimension, rot_angle)  
        return dimension 

    elif shape == "Cross":
       vertices, offset = sh.Cross()
       Vertex_Data['Cross'] = vertices
       Vertex_Data['Cross_off'] = offset
       _, _ , dimension = sh.calculate_rotated_vertex(vertices,dimension, rot_angle)  
       return dimension 

    elif shape in Vertex_Data:
       _, _ , dimension = sh.calculate_rotated_vertex(Vertex_Data[shape], dimension, rot_angle)  
       return dimension 
    
    else:
        print(str(shape), " does not exists")
        return [0, 0] 

def parse_terminal():
    global N, FREQ, SAVE_BOUNDS, DISPLAY, ADD_LIGHT_GRADIENT, OPACITY 
    global BRIGHTNESS_DIFFERENCE, BLUR_FACTOR, ADD_LETTER, KERNEL_CACHE, DEBUG
    global BLEND_MODE, NOISE_FACTOR, BRIGHTNESS, LETTER_PROB, SHAPE_SCALE 
    global SHOW_DISTRIBUTION, LIGHT_PROB, LETTER_DATASET, ONLY_LETTER
    global Background, Letters, SIZE
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n' , type = int, nargs = '?', default = N, help = 'number of images per shape') 
    parser.add_argument('--imgsz' , type = int, nargs = '?', default = 320, help = 'image size, default ={}'.format(320)) 

    parser.add_argument('--d', type = int, nargs = '?', default = int(DISPLAY), help = 'display images after a certain frequency')
    parser.add_argument('--f', type = int, nargs = '?', default = FREQ, help = 'display images after this frequency')
    parser.add_argument('--bounds', type = int, nargs = '?', default = SAVE_BOUNDS, help = 'save bounds') 
    parser.add_argument('--s0', type = float, nargs = '?', default = SHAPE_SCALE[0], help = 'shape scale lower bound, default = {}'.format(SHAPE_SCALE[0])) 
    parser.add_argument('--s1', type = float, nargs = '?', default = SHAPE_SCALE[1], help = 'shape scale upper bound, default = {}'.format(SHAPE_SCALE[1])) 
    parser.add_argument('--dist', type = float, nargs = '?', default = SHOW_DISTRIBUTION, help = 'show size distribution, default = {}'.format(SHOW_DISTRIBUTION)) 

    parser.add_argument('--noise', type = int, nargs = '?', default = NOISE_FACTOR, help = 'noise, default = {}'.format(NOISE_FACTOR)) 
    parser.add_argument('--b_diff', type = int, nargs = '?', default = BRIGHTNESS_DIFFERENCE, help = 'brighness difference') 
    parser.add_argument('--b0', type = int, nargs = '?', default = BLUR_FACTOR[0], help = 'blur lower bound, default = {}'.format(BLUR_FACTOR[0])) 
    parser.add_argument('--b1', type = int, nargs = '?', default = BLUR_FACTOR[1], help = 'blur upper bound, default = {}'.format(BLUR_FACTOR[1])) 
    # parser.add_argument('--br0', type = int, nargs = '?', default = BRIGHTNESS[0], help = 'shape brightness lower bound, default = {}'.format(BRIGHTNESS[0])) 
    # parser.add_argument('--br1', type = int, nargs = '?', default = BRIGHTNESS[1], help = 'shape brightness upper bound, default = {}'.format(BRIGHTNESS[1])) 

    parser.add_argument('--l', type = int, nargs = '?', default = int(ADD_LETTER), help = 'Add letter, default = {}'.format(ADD_LETTER)) 
    parser.add_argument('--l_prob', type = float, nargs = '?', default = LETTER_PROB, help = 'probability of a letter occuring. default: {}'.format(str(LETTER_PROB)))
    parser.add_argument('--l_dataset', type = float, nargs = '?', default = LETTER_DATASET, help = 'save datset for letter detection. default: {}'.format(str(LETTER_DATASET)))
    parser.add_argument('--only_letter', type = float, nargs = '?', default = ONLY_LETTER, help = 'only save letter dataset. default: {}'.format(str(ONLY_LETTER)))

    parser.add_argument('--debug', type = int, nargs = '?', default = int(DEBUG), help = 'will not use coloroma while running') 
    parser.add_argument('--light', type = int, nargs = '?', default = int(ADD_LIGHT_GRADIENT), help = 'add shadow in images, default = {}'.format(ADD_LIGHT_GRADIENT)) 
    parser.add_argument('--blend_prob', type = float, nargs = '?', default = LIGHT_PROB, help = 'probability of a appliying shade . default: {}'.format(str(LIGHT_PROB)))
    parser.add_argument('--opacity', type = float, nargs = '?', default = OPACITY[1], help = 'upper bound for blend opacity, (do not exceed 0.2) default = {}'.format(OPACITY)) 
    parser.add_argument('--b_mode', type = str, nargs = '?', default = BLEND_MODE, help = 'blend mode (o for overlay, s for softlight)') 

        
    args = parser.parse_args()

    if '-h' not in args and '--help' not in args:
        if os.path.exists(Output):
            shutil.rmtree(Output)
        if os.path.exists(Bounds):
            shutil.rmtree(Bounds)
        if os.path.exists(Labels):
            shutil.rmtree(Labels)
        if os.path.exists(LetterOutput):
            shutil.rmtree(LetterOutput)
        if os.path.exists(LetterBounds):
            shutil.rmtree(LetterBounds)
        if os.path.exists(LetterLabels):
            shutil.rmtree(LetterLabels)

    os.makedirs(Output, exist_ok=True)
    os.makedirs(Labels, exist_ok=True)
    os.makedirs(Bounds, exist_ok=True)
    os.makedirs(LetterOutput, exist_ok=True)
    os.makedirs(LetterLabels, exist_ok=True)
    os.makedirs(LetterBounds, exist_ok=True)

    for letter in LETTER_CLASS.keys(): 
        os.makedirs(os.path.join(LetterOutput, letter), exist_ok=True)

    if args.imgsz != 320: 

        if 'letters{}'.format(args.imgsz) not in os.listdir(os.path.join(curdir, 'res')):  
            print(f'Generating {args.imgsz}x{args.imgsz} letters')
            genl.generate_letters(args.imgsz, args.imgsz, os.path.join(curdir, 'res' , 'letters{}'.format(args.imgsz))) 
         
        if f'Backgrounds{args.imgsz}' not in os.listdir(os.path.join(curdir, '..', 'Backgrounds')):  
            print('resizing backgrounds to {}x{}')
            rb.resize_images(os.path.join(curdir, '..', 'Backgrounds', 'Backgrounds'), os.path.join(curdir, '..', 'Backgrounds', 'Backgrounds{}'.format(args.imgsz)), args.imgsz, args.imgsz )

        Background = os.path.join(curdir, "..", "Backgrounds", "Backgrounds{}".format(args.imgsz))
        Letters = os.path.join(curdir, 'res', 'letters{}'.format(args.imgsz))

    N = args.n
    DISPLAY = bool(args.d)
    FREQ = args.f
    SAVE_BOUNDS = bool(args.bounds)
    BRIGHTNESS_DIFFERENCE = max(0, min(120, args.b_diff))
    BLUR_FACTOR = [args.b0, args.b1]
    ADD_LETTER = bool(args.l) 
    LETTER_PROB = args.l_prob
    # BRIGHTNESS = [args.br0, args.br1]
    NOISE_FACTOR = args.noise
    DEBUG = bool(args.debug)
    ADD_LIGHT_GRADIENT = bool(args.light)
    SIZE = args.imgsz

    ONLY_LETTER = args.only_letter
    LETTER_DATASET = args.l_dataset
    LIGHT_PROB = args.blend_prob
    OPACITY[1] = args.opacity
    OPACITY[0] = 0.9 * OPACITY[1]
    BLEND_MODE = args.b_mode    
    SHAPE_SCALE = [args.s0, args.s1]
    SHOW_DISTRIBUTION = bool(args.dist)
    KERNEL_CACHE = {
        max(1, i): cv2.getGaussianKernel(max(1, i), 0) for i in range(BLUR_FACTOR[0], BLUR_FACTOR[1] + 1)
    }

def main():
    total = N * len(LETTER_CLASS.keys())
    f = int(0.1 * N) if FREQ == 0 else FREQ
    f = max(1, f)
    os.system('cls' if os.name == 'nt' else 'clear')

     
    print("Total images = {}".format(total))

    sh.initialize_letter_masks(Letters)
    # random.seed(time.time())
    start = time.time()
    cnt = 0
    l_cnt = 0
    shape_idx = 0
    error_lno = 6
    distribution_data = []
    try:  
        for letter in LETTER_CLASS.keys(): 
            if not DEBUG: print(colorama.Cursor.POS(0, 2) + "Letter: {}        ".format(letter))  
            else: print( "Letter: {}".format(letter))  


            for _ in range(N):
                background = random.choice(os.listdir(Background))
                while background.split(".")[-1] != "png" and background.split(".")[-1] != "jpg":
                    background = random.choice(os.listdir(Background))
                
                shape = random.choice(SHAPES)

                image = cv2.imread(os.path.join(Background, background))

                s_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] 
                l_color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                s_color, l_color = differentiate(s_color, l_color, BRIGHTNESS_DIFFERENCE)

                s_scale = random.uniform(SHAPE_SCALE[0], SHAPE_SCALE[1])
                # s_scale = SHAPE_SCALE[0]

                distribution_data.append(s_scale)
                s_scale += np.interp(s_scale, SHAPE_SCALE, SCALE_OFFSET[shape]) 

                l_scale = random.uniform(0.87, 1.1)

                s_rot_angle = random.randint(0, 360)
                l_rot_angle = random.randint(0, 360)
                
                translation = []
                if shape in ['Semicircle', 'Quartercircle']: 
                    
                    radius = (image.shape[0] * s_scale/ 2, image.shape[1] * s_scale/ 2)
                    angle_range = [s_rot_angle, s_rot_angle + 180]

                    if shape == 'Quartercircle': angle_range[1] -= 90
                    data, _= sh.circle_data(radius, radius, angle_range)

                    # translation = [
                    #     min(random.uniform(0, image.shape[0]), image.shape[0] - data['w']/2 - data['x'] ),
                    #     min(random.uniform(0, image.shape[1]), image.shape[1] - data['h']/2 - data['y']),
                    # ]
                    translation = [
                        random.uniform(- data['x'] + data['w']/2, image.shape[0] - data['w']/2 - data['x'] ),
                        random.uniform(- data['y'] + data['h']/2, image.shape[1] - data['h']/2 - data['y'])
                    ]

                elif shape != 'Circle': 

                    dimension = calculate_poly_dimension(image.shape, shape, s_scale, s_rot_angle) 
                    
                    # translation = [
                    #     min(random.uniform(0, image.shape[0]), image.shape[0] - dimension[0]),
                    #     min(random.uniform(0, image.shape[1]), image.shape[1] - dimension[1]),
                    # ]
                    translation = [
                        random.uniform(0, image.shape[0] - dimension[0]),
                        random.uniform(0, image.shape[1] - dimension[1])
                    ]

                else: 
                    dimension = [image.shape[0]* s_scale, image.shape[1]*s_scale] 
                    # translation = [
                    #     min(random.uniform(0, image.shape[0]), image.shape[0] - dimension[0]),
                    #     min(random.uniform(0, image.shape[1]), image.shape[1] - dimension[1]),
                    # ]
                    translation = [
                        random.uniform(0, image.shape[0] - dimension[0]),
                        random.uniform(0, image.shape[1] - dimension[1])
                    ]

                l_dice = random.uniform(0, 1) 
                image, data, letter_data= draw_shape(image, shape, letter, s_scale, l_scale, translation, s_rot_angle, l_rot_angle, s_color, l_color, l_dice, ADD_LETTER)

                if image is None and data is None and letter_data is None: 
                    if not DEBUG: print(colorama.Cursor.POS(0,  error_lno) + '\033[31mcannot render {}#{} \033[0m        '.format(shape, cnt))
                    error_lno += 1
                    continue

                # image = sh.bounding_box(image, data); 
                # original_scale = {
                #     'x': 0, 'y': 0, 'w': image.shape[0] * (s_scale - Scale_Offeset[shape]), 'h': image.shape[1]*(s_scale - Scale_Offeset[shape]) 
                # }
                # original_scale['x'] = data['x'] 
                # original_scale['y'] = data['y'] 
            
                # image = sh.bounding_box(image, original_scale, color = (255, 0, 0))
                
                with open(os.path.join(Labels, str(cnt) + '.txt'), "w") as label: 
                    label.write('{} {} {} {} {}'.format(shape_idx, data['x']/image.shape[0], data['y']/image.shape[1], data['w']/image.shape[0], data['h']/image.shape[1]))

                image = apply_blur(image, random.randint(BLUR_FACTOR[0], BLUR_FACTOR[1]))
                image = add_noise(image, NOISE_FACTOR)

                lighted_image = None
                if random.uniform(0, 1) < LIGHT_PROB and ADD_LIGHT_GRADIENT: 
                    light = os.path.join(Light, random.choice(os.listdir(Light)))
                    image = sh.apply_lighting_blend(image, light , random.randint(0, 360), random.uniform(OPACITY[0], OPACITY[1]), BLEND_MODE)

                # light = os.path.join(Light, random.choice(os.listdir(Light)))
                # lighted_image = sh.apply_lighting_blend(image, light , random.randint(0, 360), OPACITY[1])
                if not ONLY_LETTER: 
                    cv2.imwrite(os.path.join(Output, str(cnt) + ".jpg"), image)

                # cv2.imwrite(os.path.join(Output, str(cnt) + "_l.jpg"), lighted_image)
                
                if LETTER_DATASET and l_dice < LETTER_PROB: 
                    offset = {
                        'min' : [
                            data["x"] - (image.shape[0] * SHAPE_SCALE[1])/2 if data["x"] - (image.shape[0] * SHAPE_SCALE[1])/2 < 0 else 0, 
                            data["y"] - (image.shape[1] * SHAPE_SCALE[1])/2 if data["y"] - (image.shape[0] * SHAPE_SCALE[1])/2 < 0 else 0, 
                        ], 
                        'max' : [
                            data["x"] + (image.shape[0] * SHAPE_SCALE[1])/2 - SIZE if data["x"] + (image.shape[0] * SHAPE_SCALE[1])/2 > SIZE else 0, 
                            data["y"] + (image.shape[1] * SHAPE_SCALE[1])/2 - SIZE if data["y"] + (image.shape[0] * SHAPE_SCALE[1])/2 > SIZE else 0, 
                        ]
                    }


                    shapes_bounds = np.array([
                        [data["x"] - (image.shape[0] * SHAPE_SCALE[1])/2 - offset['min'][0] - offset['max'][0], data["y"] - (image.shape[1] * SHAPE_SCALE[1])/2- offset['min'][1] - offset['max'][1]],
                        [data["x"] + (image.shape[0] * SHAPE_SCALE[1])/2 - offset['min'][0] - offset['max'][0], data["y"] - (image.shape[1] * SHAPE_SCALE[1])/2- offset['min'][1] - offset['max'][1]],
                        [data["x"] + (image.shape[0] * SHAPE_SCALE[1])/2 - offset['min'][0] - offset['max'][0], data["y"] + (image.shape[1] * SHAPE_SCALE[1])/2- offset['min'][1] - offset['max'][1]],
                        [data["x"] - (image.shape[0] * SHAPE_SCALE[1])/2 - offset['min'][0] - offset['max'][0], data["y"] + (image.shape[1] * SHAPE_SCALE[1])/2- offset['min'][1] - offset['max'][1]],
                    ])

                    letter_image = image[int(shapes_bounds[0][1]) : int(shapes_bounds[3][1]), int(shapes_bounds[0][0]): int(shapes_bounds[1][0])]
                    # print(os.path.join(LetterOutput, letter,  str(l_cnt) + ".jpg"))
                    cv2.imwrite(os.path.join(LetterOutput, letter,  str(l_cnt) + ".jpg"), letter_image)

                    modified_letter_data = {
                        "x": (letter_data['x']-shapes_bounds[0][0]), 
                        "y": (letter_data['y']-shapes_bounds[0][1]), 
                        "w": (letter_data['w']), 
                        "h": (letter_data['h']), 
                    }

                    with open(os.path.join(LetterLabels, str(cnt) + '.txt'), "w") as label: 
                        label.write('{} {} {} {} {}'.format(
                            LETTER_CLASS[letter], 
                            modified_letter_data['x']/letter_image.shape[0], modified_letter_data['y']/letter_image.shape[1], 
                            modified_letter_data['w']/letter_image.shape[0], modified_letter_data['h']/letter_image.shape[1]))
                    
                    temp_letter_image = sh.bounding_box(letter_image, modified_letter_data) if DISPLAY or SAVE_BOUNDS else None                    
                    if SAVE_BOUNDS:  
                        cv2.imwrite(os.path.join(LetterBounds, f"{str(l_cnt)}_{letter}.jpg"), temp_letter_image)
                     
                    l_cnt += 1 

                if cnt % f == 0: 
                    # print(cli_n)
                    if not DEBUG: print(colorama.Cursor.POS(0, 3) + progress_bar(cnt, total))         
                    else: print("Images: {}".format(cnt))

                    temp_image = sh.bounding_box(image, data) if DISPLAY or SAVE_BOUNDS else None                    
                    if DISPLAY == True: 
                        # temp_image = sh.bounding_box(image,data)
                        temp_image = sh.bounding_box(temp_image, letter_data, color = (255, 255, 255))
                        cv2.imshow(str(cnt), temp_image)

                    if SAVE_BOUNDS:  
                        cv2.imwrite(os.path.join(Bounds, f"{str(cnt)}_{shape}_{letter}.jpg"), temp_image)

                    if cv2.waitKey(0) == 27:
                        run_time = time.time() - start
                        print ("{} images generated in {} minutes {} seconds".format(cnt, int(run_time/60), int((run_time - int(run_time)) * 60)))
                        cv2.destroyAllWindows()
                        return 
                    cv2.destroyAllWindows()
                cnt += 1
                        
            shape_idx += 1
    except KeyboardInterrupt:  
        print("Exiting....")        

    if SHOW_DISTRIBUTION: 
        
        sns.histplot(distribution_data, kde = True)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()

    
    run_time = time.time() - start
    cv2.destroyAllWindows()
    print ("{} images generated in {} minutes {} seconds".format(cnt, int(run_time/60), int((run_time - int(run_time)) * 60)))


if __name__ == '__main__': 
    parse_terminal()
    main()
    # test_letters_bounds(0.6)
