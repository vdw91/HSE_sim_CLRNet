import os
import json
import random
import shutil
import argparse

"""
Simple script that restructures the output from the manual labeller to one the more resembles the TUSimple dataset format.
This restructured dataset can then be used to train a lane detection model.
"""

def restructure(img_output_path, json_output_path, image_input_path):

    folder_list = os.listdir(image_input_path)

    random.shuffle(folder_list)

    train_val_test_split = [0.8, 0.1, 0.1]


    number_train = round(train_val_test_split[0] * len(folder_list))
    number_val = round(train_val_test_split[1] * len(folder_list))
    number_test = round(train_val_test_split[2] * len(folder_list))

    count = 0
    for i in range(number_train):
        
        json_path = os.path.join(image_input_path, folder_list[0], folder_list[0] +".json")
        with open(json_path) as f:
            label = json.load(f)
            
            label["raw_file"] = f"clips/number/{count}/{count}.png"
            
            os.mkdir(os.path.join(img_output_path, str(count)))
            
            shutil.copy(
                os.path.join(image_input_path, folder_list[0], f"{folder_list[0]}.png"),
                os.path.join(img_output_path, str(count), f"{count}.png")
            )
            
        with open(os.path.join(json_output_path, "train_set.json"), 'a') as out:
            out.write(json.dumps(label) + "\n")

        folder_list.pop(0)
        count += 1
        
        
        if count == number_train:
            break


    for i in range(number_val):
        
        json_path = os.path.join(image_input_path, folder_list[0], folder_list[0] +".json")
        with open(json_path) as f:
            label = json.load(f)
            
            label["raw_file"] = f"clips/number/{count}/{count}.png"

            os.mkdir(os.path.join(img_output_path, str(count)))
            
            shutil.copy(
                os.path.join(image_input_path, folder_list[0], f"{folder_list[0]}.png"),
                os.path.join(img_output_path, str(count), f"{count}.png")
            )
            
        with open(os.path.join(json_output_path, "val_set.json"), 'a') as out:
            out.write(json.dumps(label) + "\n")

        folder_list.pop(0)
        count += 1
        
        
        if count == number_val:
            break
        

    for i in range(number_test):
        
        json_path = os.path.join(image_input_path, folder_list[0], folder_list[0] +".json")
        with open(json_path) as f:
            label = json.load(f)
            
        label["raw_file"] = f"clips/number/{count}/{count}.png"
        
        os.mkdir(os.path.join(img_output_path, str(count)))
        
        shutil.copy(
            os.path.join(image_input_path, folder_list[0], f"{folder_list[0]}.png"),
            os.path.join(img_output_path, str(count), f"{count}.png")
        )
        
        with open(os.path.join(json_output_path, "test_set.json"), 'a') as out:
            out.write(json.dumps(label) + "\n")

        folder_list.pop(0)
        count += 1
        
        
        if count == number_val:
            break


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image_input_path', help='Path where manual labeller saved the data')
    parser.add_argument('img_output_path', help='Path where dataset images will be saved')
    parser.add_argument('json_output_path', help='Path where dataset JSON files will be saved (usually same as img_output_path)')

    args = parser.parse_args()

    restructure(args.img_output_path, args.json_output_path, args.image_input_path)