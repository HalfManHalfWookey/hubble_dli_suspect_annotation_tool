import cv2
import json
import os
import random
import numpy as np
from pathlib import Path
import copy

# Paths
DATA_DIR = "data/BLUE2505_11555_48876/"
IMAGE_DIR = "semseg-output/images"
ANNOTATION_FILE = "violations-output/violations.json"
SUSPECT_TO_QUERY = [5, 6, 9]
SUSPECT_NAME_MATCHING = ['TrainContact', 'WorkerSighting', 'VegEncroaching']

# Load annotations
with open(os.path.join(DATA_DIR, ANNOTATION_FILE), 'r') as f:
    violations_json = json.load(f)

# Sort out violations into a better format
annotations = {}
for frame, entries in violations_json.items():
    if 'violations' in entries:
        violations = []
        for violation in entries['violations']: 
                if violation['group'] == 0:
                    continue
                
                if len(violation['suspects'])==0:
                    continue

                for suspect in violation['suspects']:
                    violation['suspect'] = suspect
                    violation['subtype'] = SUSPECT_NAME_MATCHING[SUSPECT_TO_QUERY.index(suspect['suspect_type'])]
                    violation['positive'] = False
                    violation['manual'] = False
                    violations.append(violation)

        annotations[frame] = violations

# Get sorted list of image filenames for frame indexing
image_filenames = sorted([f for f in os.listdir(os.path.join(DATA_DIR, IMAGE_DIR))])

# Convert normalized bbox to absolute pixel coordinates
def denormalize_bbox(bbox, img_w, img_h):
    x_center, y_center, width, height = bbox
    x = int((x_center - width / 2) * img_w)
    y = int((y_center - height / 2) * img_h)
    w = int(width * img_w)
    h = int(height * img_h)
    return (x, y, w, h)

# Convert pixel bbox to normalised
def normalize_bbox(bbox, img_h, img_w):
    cx, cy, w, h = bbox
    return (cx / img_w, cy / img_h, w / img_w, h / img_h)

# Draw bounding boxes
def show_image_with_boxes(img, boxes, color=(0, 255, 0), label="GT"):
    img_copy = img.copy()
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img_copy, label, (x, int(y + (h/2) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img_copy

# Global state for click
selected_click = None

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, box, img, img_copy

    ix, iy = x, y


# Find next recommended frame
def find_next_recommended_frame(current_index, annotations, filenames):
    num_files = len(filenames)

    for offset in range(1, num_files):
        # Search forward
        if current_index + offset < num_files:
            forward_name = filenames[current_index + offset]
            try:
                if annotations.get(f'frame_{forward_name.split('.')[0]}'):
                    for i, entry in enumerate(annotations.get(f'frame_{forward_name.split('.')[0]}')):
                        if entry['recommended'] == True:
                            return current_index + offset
            except KeyError as e:
                continue

    return current_index


def find_prev_recommended_frame(current_index, annotations, filenames):
    num_files = len(filenames)

    for offset in range(1, num_files):
        # Search backward
        if current_index - offset >= 0:
            backward_name = filenames[current_index - offset]
            try:
                if annotations.get(f'frame_{backward_name.split('.')[0]}'):
                    for i, entry in enumerate(annotations.get(f'frame_{backward_name.split('.')[0]}')):
                        if entry['recommended'] == True:
                            return current_index - offset
            except KeyError as e:
                continue

    return current_index

def generate_manual_entry(suspect_type, x, y):
    new_violation = {}
    new_violation['subtype'] = suspect_type
    new_violation['suspects'] = []
    new_violation['suspects'].append({'asset_type': 6, 'suspect_type': SUSPECT_TO_QUERY[SUSPECT_NAME_MATCHING.index('TrainContact')]})
    new_violation['cx'],  new_violation['cy'],  new_violation['w'],  new_violation['h'] = normalize_bbox((x, y, 100, 100), img_h, img_w)
    new_violation['manual'] = True
    new_violation['positive'] = True
    new_violation['recommended'] = True
    new_violation['group'] = 'M'
    return new_violation

cv2.namedWindow('Output')
cv2.setMouseCallback("Output", draw_rectangle)

# Main loop
current_index = 0
prev_index = -1
while True:
    image_name = image_filenames[current_index]
    image_path = os.path.join(IMAGE_DIR, image_name)
    img = cv2.imread(os.path.join(DATA_DIR, image_path))

    if img is None:
        continue

    img_h, img_w = img.shape[:2]

    # Show image to click
    img_display = img.copy()
    cv2.imshow('Output', img_display)

    
    # if prev_index != current_index:
    try:
        violation_entry = annotations[f'frame_{image_name.split('.')[0]}']

        gt_img_display = copy.deepcopy(img_display)
        for i, violation in enumerate(violation_entry):
            gt_h, gt_w = gt_img_display.shape[:2]
            if violation['manual']:
                label = f'{violation['subtype']}_{violation['group']}'
                colour=(0, 0, 255)
            elif violation['recommended'] == True and violation['manual'] != True:
                if violation['positive']:
                    colour=(255, 165, 0)
                else:
                    colour=(0, 165, 255)
                label = f'{violation['subtype']}_{violation['group']}_({i+1})'
            else:
                colour=(165, 165, 0)
                label = f'{violation['subtype']}_{ violation['group']}'
            gt_img_display = show_image_with_boxes(gt_img_display, [denormalize_bbox((violation['cx'], violation['cy'], violation['w'], violation['h']), gt_w, gt_h)], color=colour, label=f'{label}')
        cv2.imshow('Output', gt_img_display)
        key = cv2.waitKey(0)
    except KeyError:
        cv2.imshow('Output', img_display)
        key = cv2.waitKey(0)

    # prev_index = current_index
    
    if key >= 49 and key <=57:
        index = key-49
        try:
            annotations[f'frame_{image_name.split('.')[0]}'][index]['positive'] = not(annotations[f'frame_{image_name.split('.')[0]}'][index]['positive'])
        except KeyError:
            continue
        except IndexError:
            continue

    if(key == 0):
        current_index = find_next_recommended_frame(current_index, annotations, image_filenames)

    if(key == 3):
        current_index = np.min((current_index + 1, len(image_filenames)))

    if (key == 2):
        current_index = np.max((current_index - 1, 0))

    if(key == 1):
        current_index = find_prev_recommended_frame(current_index, annotations, image_filenames)

    #U - train contact
    if(key == 117):
        new_violation = generate_manual_entry('TrainContact', ix, iy)
        annotations[f'frame_{image_name.split('.')[0]}'].append(new_violation)

    #I - WorkerSighting
    if(key == 105):
        new_violation = generate_manual_entry('WorkerSighting', ix, iy)
        annotations[f'frame_{image_name.split('.')[0]}'].append(new_violation)

    #P - vegencroaching
    if(key == 112):
        new_violation = generate_manual_entry('VegEnchroachment', ix, iy)
        annotations[f'frame_{image_name.split('.')[0]}'].append(new_violation)

    with open(os.path.join(DATA_DIR, 'violations-output/violations-annotated.json'), 'w') as f:
        json.dump(annotations, f)

    