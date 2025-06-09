import cv2
import json
import os
import random
import numpy as np
from pathlib import Path
import copy
import argparse



class annotator:
    def __init__(self, args):
        self.DATA_DIR = args.dir
        self.IMAGE_DIR = "semseg-output/images"
        self.ANNOTATION_FILE = "violations-output/violations.json"
        self.SUSPECT_TO_QUERY = [5, 6, 9]
        self.SUSPECT_NAME_MATCHING = ['TrainContact', 'WorkerSighting', 'VegEncroaching']

        # Load annotations
        with open(os.path.join(self.DATA_DIR, self.ANNOTATION_FILE), 'r') as f:
            self.violations_json = json.load(f)

        # Sort out violations into a better format
        self.annotations = {}
        for frame, entries in self.violations_json.items():
            if 'violations' in entries:
                violations = []
                for violation in entries['violations']: 
                        if violation['group'] == 0:
                            continue
                        
                        if len(violation['suspects'])==0:
                            continue

                        for suspect in violation['suspects']:
                            violation['suspect'] = suspect
                            violation['subtype'] = self.SUSPECT_NAME_MATCHING[self.SUSPECT_TO_QUERY.index(suspect['suspect_type'])]
                            violation['positive'] = False
                            violation['manual'] = False
                            violations.append(violation)

                self.annotations[frame] = violations

        # Get sorted list of image filenames for frame indexing
        self.image_filenames = sorted([f for f in os.listdir(os.path.join(self.DATA_DIR, self.IMAGE_DIR))])

    # Convert normalized bbox to absolute pixel coordinates
    def denormalize_bbox(self, bbox, img_w, img_h):
        x_center, y_center, width, height = bbox
        x = int((x_center - width / 2) * img_w)
        y = int((y_center - height / 2) * img_h)
        w = int(width * img_w)
        h = int(height * img_h)
        return (x, y, w, h)

    # Convert pixel bbox to normalised
    def normalize_bbox(self, bbox, img_h, img_w):
        cx, cy, w, h = bbox
        return (cx / img_w, cy / img_h, w / img_w, h / img_h)

    # Draw bounding boxes
    def show_image_with_boxes(self, img, boxes, color=(0, 255, 0), label="GT"):
        img_copy = img.copy()
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img_copy, label, (x, int(y + (h/2) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img_copy

    def get_mouse_position(self, event, x, y, flags, param):
        self.ix, self.iy = x, y


    # Find next recommended frame
    def find_next_recommended_frame(self, current_index, annotations, filenames):
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


    def find_prev_recommended_frame(self, current_index, annotations, filenames):
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

    def generate_manual_entry(self, suspect_type, x, y, img_h, img_w):
        new_violation = {}
        new_violation['subtype'] = suspect_type
        new_violation['suspects'] = []
        new_violation['suspects'].append({'asset_type': 6, 'suspect_type': self.SUSPECT_TO_QUERY[self.SUSPECT_NAME_MATCHING.index('TrainContact')]})
        new_violation['cx'],  new_violation['cy'],  new_violation['w'],  new_violation['h'] = self.normalize_bbox((x, y, 100, 100), img_h, img_w)
        new_violation['manual'] = True
        new_violation['positive'] = True
        new_violation['recommended'] = True
        new_violation['group'] = 'M'
        return new_violation

    def main(self):
        cv2.namedWindow('Output')
        cv2.setMouseCallback("Output", self.get_mouse_position)

        # Main loop
        current_index = 0
        prev_index = -1
        while True:
            image_name = self.image_filenames[current_index]
            image_path = os.path.join(self.IMAGE_DIR, image_name)
            img = cv2.imread(os.path.join(self.DATA_DIR, image_path))

            if img is None:
                continue

            img_h, img_w = img.shape[:2]

            # Show image to click
            img_display = img.copy()
            cv2.imshow('Output', img_display)

            
            # if prev_index != current_index:
            try:
                violation_entry = self.annotations[f'frame_{image_name.split('.')[0]}']

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
                    gt_img_display = self.show_image_with_boxes(gt_img_display, [self.denormalize_bbox((violation['cx'], violation['cy'], violation['w'], violation['h']), gt_w, gt_h)], color=colour, label=f'{label}')
                cv2.imshow('Output', gt_img_display)
                key = cv2.waitKey(0)
            except KeyError:
                cv2.imshow('Output', img_display)
                key = cv2.waitKey(0)

            # 1-9: Set bounding box to positive/negatiive
            if key >= 49 and key <=57:
                index = key-49
                try:
                    self.annotations[f'frame_{image_name.split('.')[0]}'][index]['positive'] = not(self.annotations[f'frame_{image_name.split('.')[0]}'][index]['positive'])
                except KeyError:
                    continue
                except IndexError:
                    continue

            # Up: go to next recommended frame
            if(key == 0):
                current_index = self.find_next_recommended_frame(current_index, self.annotations, self.image_filenames)

            # Right: go to next frame
            if(key == 3):
                current_index = np.min((current_index + 1, len(self.image_filenames)))

            # Left: Go to previous frame
            if (key  == 2):
                current_index = np.max((current_index - 1, 0))

            # Down: Go to the prevuous recommended frame
            if(key == 1):
                current_index = self.find_prev_recommended_frame(current_index, self.annotations, self.image_filenames)

            # U - train contact
            if(key == 117):
                new_violation = self.generate_manual_entry('TrainContact', self.ix, self.iy, img_h, img_w)
                self.annotations[f'frame_{image_name.split('.')[0]}'].append(new_violation)

            # I - WorkerSighting
            if(key == 105):
                new_violation = self.generate_manual_entry('WorkerSighting', self.ix, self.iy, img_h, img_w)
                self.annotations[f'frame_{image_name.split('.')[0]}'].append(new_violation)

            # O - vegencroaching
            if(key == 111):
                new_violation = self.generate_manual_entry('VegEnchroachment', self.ix, self.iy, img_h, img_w)
                self.annotations[f'frame_{image_name.split('.')[0]}'].append(new_violation)
            
            # Save outputs after we might have changed something before we quit
            with open(os.path.join(self.DATA_DIR, 'violations-output/violations-annotated.json'), 'w') as f:
                json.dump(self.annotations, f)

            #ESC - Quit
            if(key==27):
                quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GUi tool for generating metrics for hubble DLI suspect types')
    parser.add_argument('--dir', help='Dataset diretcory (default hubble output formats)', required=True)
    parser.add_argument('--output-file', help='Name of the output file', default='violations-annotated.json')
    args = parser.parse_args()
    
    annotate = annotator(args)
    annotate.main()