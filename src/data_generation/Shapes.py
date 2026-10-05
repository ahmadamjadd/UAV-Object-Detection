import cv2
import random
import os
import numpy as np
from functools import lru_cache
Letter_Mask = {}
    
def initialize_letter_masks(letter_dir):  
    for l_name in os.listdir(letter_dir):
        letter = cv2.imread(os.path.join(letter_dir, l_name))
        # hsv_image = cv2.cvtColor(letter, cv2.COLOR_BGR2HSV)
        Letter_Mask[l_name[0]] =  cv2.inRange(letter, np.array([0, 0, 0]), np.array([0, 0, 0]))

def apply_lighting_blend(image, light_path, angle, opacity = 0.4, mode = 's'): 

    # light = cv2.imread(light_path, 1)
    # if image.shape[2] > 3: image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)    

    # rotation_matrix = cv2.getRotationMatrix2D((light.shape[0] / 2, light.shape[1]/2), angle, 1)

    # light = cv2.warpAffine(light, rotation_matrix, (light.shape[0], light.shape[1]))
    # x_min = (light.shape[0] - image.shape[0])//2
    # x_max = x_min + image.shape[0]

    # y_min = (light.shape[1] - image.shape[1])//2
    # y_max = y_min + image.shape[1]

    # light = light[x_min:x_max, y_min:y_max]
    # black = cv2.inRange(light, np.array([0, 0, 0]), np.array([0, 0, 0]))

    # light = cv2.cvtColor(light, cv2.COLOR_BGR2BGRA)
    # light[:, :, 3] = 255 - black

    # light = light.astype(float)/255 # make float on range 0-1

    # ab = np.zeros_like(new_image) 
    # if mode == 'o':
    #     mask = (new_image>= 0.5)   
    #     # now do the blending 
    #     ab[~mask] = (2*new_image*light)[~mask] 
    #     ab[mask] = (1-2*(1-new_image)*(1-light))[mask] 
    # elif mode == 's':
    #     mask = (light >= 0.5)
    #     ab[~mask] = (2*new_image*light + new_image*new_image*(1-2*light))[~mask] 
    #     ab[mask] = (2*new_image*(1-light) + pow(new_image, 0.5)*(2*light - 1))[mask] 

    # ab =(ab*255).astype(np.uint8) 
    # result = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_BGR2BGRA), 1- opacity, light, opacity, 0) 
    # return cv2.cvtColor(result, cv2.COLOR_BGRA2BGR) 

    height, width = image.shape[:2]
    gradient_mask = np.zeros((height + 200, width + 200), dtype=np.uint8)

    for x in range(gradient_mask.shape[0]):
        gradient_mask[:, x] = int(255 * (x / gradient_mask.shape[0]))

    gradient_mask = cv2.cvtColor(gradient_mask, cv2.COLOR_GRAY2BGR)

    rotation_matrix = cv2.getRotationMatrix2D((gradient_mask.shape[0] / 2, gradient_mask.shape[1]/2), angle, 1)
    gradient_mask = cv2.warpAffine(gradient_mask, rotation_matrix, (gradient_mask.shape[0], gradient_mask.shape[1]))
    x_min = (gradient_mask.shape[0] - image.shape[0])//2
    x_max = x_min + image.shape[0]

    y_min = (gradient_mask.shape[1] - image.shape[1])//2
    y_max = y_min + image.shape[1]

    gradient_mask = gradient_mask[x_min:x_max, y_min:y_max]

    return cv2.addWeighted(image, 1.0, gradient_mask, opacity, 0.0)
    

def blend_shapes(shaped_image, background, color, vertices, alpha = 0.5): 
    mask = cv2.fillConvexPoly(foreground, [vertices], color = (0, 0, 0))
    mask = cv2.inRange(mask, np.array([0, 0, 0]), np.array[0, 0, 0])

    foreground = np.ones_like(background, np.uint8) * 255
    foreground = cv2.fillPoly(foreground, [vertices], color = color)    

    foreground = foreground.astype(float)/255
    background = background.astype(float)/255

    overlay = np.zeros_like(background)
    overlay_condition = background >= 0.5 
    overlay[~overlay_condition] = (2 * foreground * background)[~overlay_condition]
    overlay[overlay_condition] = (1-2*(1-background)*(1-foreground))[overlay_condition] # else this
    overlay =(overlay*255).astype(np.uint8) 

    shaped_image[mask == 255] = cv2.addWeighted(shaped_image[mask == 255], alpha, overlay[mask == 255], (1 - alpha))

    return shaped_image
    


def add_letter(background , letter_name, color = (0, 0, 255), scale = 1, translation = [0, 0], rotation = 0, test = False):

    
    mask = Letter_Mask[letter_name]
 
    rows, cols = mask.shape
    M_rotation = cv2.getRotationMatrix2D((cols/ 2, rows / 2), rotation, 1)
    mask = cv2.warpAffine(mask, M_rotation, (cols, rows), flags= cv2.INTER_NEAREST)
    scale *= 1.2 * (background.shape[0]/mask.shape[0])
    mask = cv2.resize(mask, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    
    translation = np.array([translation[0] - background.shape[0] * scale / 2 , translation[1] - background.shape[1] * scale / 2], np.int32)
    y1, y2 = translation[1], translation[1] + mask.shape[1] 
    x1, x2 = translation[0], translation[0] + mask.shape[0] 

    y1 = max(0, y1)
    y2 = min(background.shape[1], y2)
    x1 = max(0, x1)
    x2 = min(background.shape[0], x2)

    white_pixels = np.where(mask == 255)

    mask =  mask[0:y2 - y1, 0:x2 - x1] 
    background[y1:y2, x1:x2][white_pixels] = color
    # print(np.min(white_pixels[0]))
    
    min_values = [np.min(white_pixels[i]) for i in range(1, -1, -1)]
    max_values = [np.max(white_pixels[i]) for i in range(1, -1, -1)]
    dimension = [ max_values[i] - min_values[i] for i in range(2)]

    data = {
        "x": min_values[0] + dimension[0]/2 + translation[0], 
        "y": min_values[1] + dimension[1]/2 + translation[1], 
        "w": dimension[0], 
        "h": dimension[1]
    } 
    return background, data 

def circle_data(center, radius, angle_range):
    if (angle_range[1] - angle_range[0] == 360):
        return {
            "x" : center[0], 
            "y" : center[1], 
            "w" : 2 * radius[0], 
            "h" : 2 * radius[0]
        }, center
    
    edge_pos = [
        [center[0] + radius[0] * np.cos(np.deg2rad(angle_range[0])), center[1] + radius[0] * np.sin(np.deg2rad(angle_range[0]))],
        [center[0] + radius[0] * np.cos(np.deg2rad(angle_range[1])), center[1] + radius[0] * np.sin(np.deg2rad(angle_range[1]))],
        [center[0], center[1]] 
    ]
    # for i in range (0, 3):
    #     image = cv2.circle(image, (int(edge_pos[i][0]), int(edge_pos[i][1])), 5, (255, 0, 0))
    angle_r = range(angle_range[0], angle_range[1] + 1)
     
    x = {
        "min": min(center[0] - (radius[0] if 180 in angle_r else 0),
                    min(edge_pos, key = lambda x: x[0])[0]),

        "max": max(center[0] + (radius[0] if 0 in angle_r or 360 in angle_r else 0)
                   , max(edge_pos, key = lambda x: x[0])[0]) 
    } 
    y = {
        "min": min(center[1] - (radius[0] if 270 in angle_r else 0)
                   , min(edge_pos, key = lambda x: x[1])[1]),

        "max": max(center[1] + (radius[0] if 90 in angle_r or 450 in angle_r else 0)
                   , max(edge_pos, key = lambda x: x[1])[1]) 
    }
    
    dimension = [
        x["max"] - x["min"], 
        y["max"] - y["min"]
    ]
    semi_centeroid_offset = 2 if angle_range[1] - angle_range[0] == 180 else 1
    edge_pos.append(
        [center[0] + semi_centeroid_offset * radius[0] * np.cos(np.deg2rad(np.average(angle_range))) , 
         center[1] + semi_centeroid_offset * radius[0] * np.sin(np.deg2rad(np.average(angle_range))) ],
    )
    edge_pos = np.array(edge_pos)

    centroid = [
        np.mean(edge_pos[:, 0]), 
        np.mean(edge_pos[:, 1])
    ]

    return {
        "x" : x["min"] + dimension[0]/2, 
        "y" : y["min"] + dimension[1]/2, 
        "w" : dimension[0],
        "h" : dimension[1]
    }, centroid

def bounding_box(image, data, width = 1, color = (0, 0, 255)):
    vertex = np.array([
        [data["x"] - data["w"]/2, data["y"] - data["h"]/2],
        [data["x"] + data["w"]/2, data["y"] - data["h"]/2],
        [data["x"] + data["w"]/2, data["y"] + data["h"]/2],
        [data["x"] - data["w"]/2, data["y"] + data["h"]/2],
    ], np.int32)

    # image = cv2.circle(image, (int(data["x"]), int(data["y"])), 2, -1)
    return cv2.polylines(image, [vertex], True, color , width)

def Circle(image, angle_range, scale, color, center, border):
    radius = (image.shape[0] * scale / 2, image.shape[1] * scale / 2)
    # center = (dimension[0]/(image.shape[0] * 2) + translation[0] / image.shape[0], dimension[1]/(image.shape[1] * 2)+ translation[1] / image.shape[1])
    center = [center[0] + radius[0] , center[1] + radius[1]]

    if angle_range[0] == 0 and angle_range[1] == 360:
        cv2.circle(image, (int(center[0]), int(center[1])), int(radius[0]), color, -1)
        data, centroid = circle_data(center, radius, angle_range)
        return image, data, centroid 

    axes = (int(radius[0]), int(radius[1]))
    cv2.ellipse(image, (int(center[0]), int(center[1])), axes, 0, angle_range[0], angle_range[1], color, border if border else -1)

    data, centroid = circle_data(center, radius, angle_range)
    return image, data, centroid
    
# @lru_cache(maxsize = None)
def calculate_rotated_vertex(vertices, dimension , rot_angle):
    rot_angle = np.deg2rad(rot_angle)
    center = (0.5 * dimension[0], 0.5 * dimension[1])
    scaled_vertices = np.array([[vertex[0] * dimension[0] - center[0], vertex[1] * dimension[1] - center[1]] for vertex in vertices], np.int32)

    rotation_matrix = np.array([
        [np.cos(rot_angle), -np.sin(rot_angle)],
        [np.sin(rot_angle), np.cos(rot_angle)]
    ])

    rotated_vertices = np.array([np.matmul(rotation_matrix , vertex) for vertex in scaled_vertices], np.int32)
    min_vertex = [
       min(rotated_vertices, key= lambda x: x[0])[0],
       min(rotated_vertices, key= lambda x: x[1])[1]
    ]
    max_vertex = [
       max(rotated_vertices, key= lambda x: x[0])[0],
       max(rotated_vertices, key= lambda x: x[1])[1]
    ]

    dimension = [
        max_vertex[0] - min_vertex[0],
        max_vertex[1] - min_vertex[1],
    ]
    return min_vertex , rotated_vertices, dimension

def Polygon(image, vertices, dimension, color, rot_angle, translation, s_border):


    min_vertex, rotated_vertices, rotated_dimension = calculate_rotated_vertex(vertices, dimension, rot_angle) 


    translated_vertices = np.array([[vertex[0] + abs(min_vertex[0]) + translation[0], vertex[1] + abs(min_vertex[1]) + translation[1]] for vertex in rotated_vertices], np.int32)
    # translated_vertices = tuple(tuple(vertex) for vertex in translated_vertices)
    if s_border: 
        image = cv2.polylines(image, [translated_vertices], isClosed=True, color=(0, 0, 0), thickness=s_border)
    image = cv2.fillPoly(image, [translated_vertices], color = color)

    #calculating the dimensions after rotation for the bounding box
    centroid = [
        np.mean(translated_vertices[:, 0]), 
        np.mean(translated_vertices[:, 1])
    ] 

    data = {
        "x" : translation[0] + rotated_dimension[0]/2,
        "y" : translation[1] + rotated_dimension[1]/2,
        "w" : rotated_dimension[0],
        "h" : rotated_dimension[1]
    }
    return image, data, centroid


def Cross():
    vertices = []
    center = (0.5, 0.5)
    # offset = np.random.normal(loc = 0.12, scale = 0.1, size = 1)[0]
    # offset = min(0.14, max(0.12, offset))
    offset = random.uniform(1/6, 1/3) / 2   
    cross_end = [[1.0, 0.5], [0.5, 0.0], [0.0, 0.5], [0.5, 1.0]]
    center_x = [ 1, -1,  -1, 1]
    center_y = [ -1, -1, 1, 1]
    arm_x = [0, 0, +1 , -1, 0, 0, -1, 1]
    arm_y = [1, -1, 0, 0, -1, 1 , 0, 0]

    j_count = 0
    for i in range(4):
        for j in range(j_count , j_count + 2):
            vertices.append([cross_end[i][0] + arm_x[j] * offset, cross_end[i][1] + arm_y[j] * offset])
        vertices.append([center[0] + center_x[i] * offset, center[1] + center_y[i] * offset])
        j_count = j_count + 2

    return vertices, offset * 2    

def Rectangle(s_scale):
    vertices = []
    center = (0.5, 0.5)
    # offset = [random.uniform(0.35, 0.5), random.uniform(0.35, 0.5)]
    # if offset[0] < offset[1]:
    #    offset.reverse()

    offset = [random.uniform(0.15, 0.3)]
    offset = [s_scale/2]
    offset.append(random.uniform(offset[0] + 0.1, 0.425))

    x = [1, -1, -1, 1]
    y = [1, 1, -1, -1]
    for i in range(4):
        vertices.append([center[0]+ x[i] * offset[0], center[0]+ y[i] * offset[1]])
    return vertices, min(offset) * 2

def Trapezoid():
    offset = random.uniform(0.15, 0.25)
    vertices = [
        [0.5 + offset, 0.15], 
        [0.5 - offset, 0.15], 
        [0.15, 0.85], 
        [0.85, 0.85]
    ]
    return vertices