import cv2
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line
import tkinter as tk
import os
import glob
import shutil
import json

    
class LabelTool(tk.Frame):
    """
    Tool to manuzally label images according to the TuSimple dataset format.
    The GUI displays the image along with the h_samples which will be used to store the lane positions.
    
    To draw a lane, left click on the image to set the start point of the line, then click again to set the end point.
    While drawing a lane, keep clicking to add another point, continuing the line from the previous point.
    In case of a mistake, backspace will undo the last drwan line (and only the last line).   
    When a wrong starting point has been selected, the delete key will reset the postion. 
    
    Start with lane 0, which is the middle lane.
    Afterwards press right click to swtich to lane 1, which is right of the car.
    Finally press right click again to switch to lane 2, which is left of the car.
    """
    
    def __init__(self, parent, img_dir_path, output_dir_path):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        
        self.w = tk.Canvas(root, width=1300, height=1300)
        self.w.pack()
        
        # retrieve all to be labelled images as an iterator
        self.img_iterator = glob.iglob(img_dir_path + "/*.png")
        
        # Check the output folder for last numbered image, to continue where stopped
        self.count = 0
        last_labelled_check = os.listdir(output_dir_path)
    

        last_labelled_check = [int(x) for x in last_labelled_check]
        last_labelled_check.sort()
        

        if len(last_labelled_check) >= 1:
            for i in range(int(last_labelled_check[-1]) + 1):
                self.count += 1
                next(self.img_iterator)


        self.labelled = last_labelled_check

        self.img_path = next(self.img_iterator)
        img = cv2.imread(self.img_path)

        # resize the image so its easier to be acurate while labelling
        self.img = cv2.resize(img, (1300, 1300)) 

        # draw line on half the image since that is the limit of what should be labelled,
        # lanes get blurry afterwards
        # cv2.line(self.img, (0, 700), (1300, 700), (0, 255 ,0), thickness=2)
        self.draw_guidelines()
        
        # keep a copy of the image for a undo function
        self.undo_image = self.img.copy()
        
        # display the image
        image =  ImageTk.PhotoImage(image=Image.fromarray(self.img))
        self.w.imgref = image
        self.image_on_canvas = self.w.create_image(0, 0, image=image, anchor="nw")
        
        self.output_dir_path = output_dir_path
        
        # Create an array to store the actual labels
        self.output_array = np.full((800,800), 255)

        self.prev_x = 0
        self.prev_y = 0
        self.draw = False
        
        self.lane = 0
        self.lane_colours = [(255, 0, 0), (0,255, 0),(0, 0, 255)]

        self.w.bind("<Button 1>", self.getorigin)
        self.w.bind("<Button 3>", self.cancel_draw)
        root.bind("<BackSpace>", self.undo)
        root.bind("<Return>", self.next_image)
        root.bind("<Delete>", self.pos_reset)

        print(self.count)

    def draw_output(self, x1, y1, x2, y2):
        # keep a copy for the undo function
        self.undo_output = self.output_array.copy()
        
        
        # use skimage.draw line function to draw a line in the array with the number of the lane
        rr, cc = line(y1, x1, y2, x2)
        self.output_array[rr, cc] = self.lane
        
        # plt.imshow(self.output_array)
        # plt.show(block=False)

        

    def getorigin(self, eventorigin):
        
        # get click pos
        self.x0 = eventorigin.x
        self.y0 = eventorigin.y    
        
        self.undo_x = self.prev_x
        self.undo_y = self.prev_y
        
        if self.draw == True:
            self.undo_image = self.img.copy()
            cv2.line(self.img, (self.prev_x, self.prev_y), (self.x0, self.y0), self.lane_colours[self.lane], thickness=2)
            self.draw_output(LabelTool.rel_to_abs_pos(self.prev_x), LabelTool.rel_to_abs_pos(self.prev_y), LabelTool.rel_to_abs_pos(self.x0), LabelTool.rel_to_abs_pos(self.y0))
            
            self.prev_x = self.x0
            self.prev_y = self.y0
            
            self.draw = True

        else:
            self.draw = True
            self.prev_x = self.x0
            self.prev_y = self.y0
           
        # update image with new lines 
        image =  ImageTk.PhotoImage(image=Image.fromarray(self.img))
        self.w.imgref = image
        self.w.itemconfig(self.image_on_canvas, image = image)

            
    def cancel_draw(self, eventorigin):
        # restart line when switching lanes
        self.draw = None
        self.lane += 1
        
    def pos_reset(self, event):
        self.prev_x = 0
        self.prev_y = 0
        self.draw = False

    def undo(self, event):

        # reset evertything to the saves from previous action
        image =  ImageTk.PhotoImage(image=Image.fromarray(self.undo_image))
        self.w.imgref = image
        self.w.itemconfig(self.image_on_canvas, image = image)
        self.img = self.undo_image
        
        self.prev_x = self.undo_x
        self.prev_y = self.undo_y
        
        self.output_array = self.undo_output
        
        
    def next_image(self, event):
        self.save_output()
        

        
        self.img_path = next(self.img_iterator)
        img = cv2.imread(self.img_path)
        self.img = cv2.resize(img, (1300, 1300)) 
        
        # cv2.line(self.img, (0, 700), (1400, 700), (0, 255 ,0), thickness=2)
        self.draw_guidelines()
        self.undo_image = self.img.copy()
        image =  ImageTk.PhotoImage(image=Image.fromarray(self.img))
        self.w.imgref = image
        self.image_on_canvas = self.w.create_image(0, 0, image=image, anchor="nw")
        
        self.prev_x = 0
        self.prev_y = 0
        self.draw = False
        
        self.lane = 0
        

    def draw_guidelines(self):
        cv2.line(self.img, (0, 650), (1300, 650), (0, 255 ,0), thickness=1)  
        
        for hline in range(400, 800, 10):
            scaled_hline = round((hline / 800) * 1300)
            # print(scaled_hline)
            cv2.line(self.img, (0, scaled_hline), (1300, scaled_hline), (0, 255 ,0), thickness=1)  
        
        
    def save_output(self):
        #create a new folder and add a copy of the labelled image
        os.mkdir(os.path.join(self.output_dir_path, str(self.count)))
        shutil.copy(self.img_path, os.path.join(self.output_dir_path, str(self.count), f"{self.count}.png"))
        
        output_json = {
        "lanes": [],
        "h_samples" : [],
        }
        
        lanes = {
            "lane_0": [],
            "lane_1": [],
            "lane_2": [],
        }
        
        # iterate over the output array to create a dataset in the TUsimple fomat
        for hline in range(400, 800, 10):
            output_json["h_samples"].append(hline)
            for lane in range(3):
                for i, value in enumerate(self.output_array[hline]):
                    if value == lane:
                        lanes[list(lanes)[lane]].append(i)
                        break
                else: 
                    lanes[list(lanes)[lane]].append(-2)
              
              
        output_json["lanes"].append(lanes["lane_0"])        
        output_json["lanes"].append(lanes["lane_1"])        
        output_json["lanes"].append(lanes["lane_2"])        
                    
        with open(os.path.join(self.output_dir_path, str(self.count), f"{self.count}.json"), 'w') as f:
            json.dump(output_json, f)
            
        self.output_array = np.full((800,800), 255) 
        self.count += 1
            

    @staticmethod
    def rel_to_abs_pos(pos):
        # calculate the coos of click to the size of the original image
        return round((pos / 1300) * 800)     


if __name__ == "__main__":
    
    img_dir_path = r"imgs" # Path to a folder with images to be labelled
    output_dir_path = r"labelled_data" # Path to a folder where the labelled data will be saved
    
    root = tk.Tk()
    app = LabelTool(root, img_dir_path=img_dir_path, output_dir_path=output_dir_path)
    root.mainloop()