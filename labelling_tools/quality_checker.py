
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import os
import json
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)


class LabelTool(tk.Frame):
    """
    Very simple tool that iterates over every labelled image provided (assumed to be labelled using the accompanying manual labeller).
    The user gets the option to keep the displayed image and label by pressing the Enter key, or delete the image and label by pressing the Delete key.
    """
    
    def __init__(self, parent, output_dir_path):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.output_dir_path = output_dir_path
        
        folders =  os.listdir(output_dir_path)
        folders = [int(x) for x in folders]

        folders.sort()
        self.folder_list = iter(folders)

        # the figure that will contain the plot
        fig = Figure(figsize = (1600, 1600), dpi= 5)

        fig, self.ax = plt.subplots()
        self.root = root

        self.canvas = FigureCanvasTkAgg(fig,
                        master = root)  
    
        # placing the canvas on the Tkinter window
        self.canvas.get_tk_widget().pack()
    
        # creating the Matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas,
                                    root)
        self.toolbar.update()
    
        # placing the toolbar on the Tkinter window
        self.canvas.get_tk_widget().pack()
    
        root.bind("<Return>", self.plot_next)
        root.bind("<Delete>", self.delete_data)
            
            
    def plot_next(self, event):
        
        self.current_folder = next(self.folder_list)
        print(self.current_folder)
        img = cv2.imread(os.path.join(self.output_dir_path, str(self.current_folder), str(self.current_folder) + ".png"))

        json_path = os.path.join(self.output_dir_path, str(self.current_folder),  str(self.current_folder) + ".json")
        with open(json_path) as f:
            label = json.load(f)

        self.ax.clear()
        
        fig = Figure(figsize = (800, 800),
                    dpi = 100)
    
        im = self.ax.imshow(img)
        
        h_samples = np.array([int(x) for x in label["h_samples"]])
        lane_0 = np.array([int(x) for x in label["lanes"][0]])
        lane_1 = np.array([int(x) for x in label["lanes"][1]])
        lane_2 = np.array([int(x) for x in label["lanes"][2]])
    
        mask_0 = lane_0 >= 0
        lane_0 = lane_0[mask_0]
        h_0 = h_samples[mask_0]
        
        mask_1 = lane_1 >= 0
        lane_1 = lane_1[mask_1]
        h_1 = h_samples[mask_1]
        
        mask_2 = lane_2 >= 0
        lane_2 = lane_2[mask_2]
        h_2 = h_samples[mask_2]
            
        plt.plot(lane_0, h_0, color="red")
        plt.plot(lane_1, h_1, color="blue")
        plt.plot(lane_2, h_2, color="green")

        self.canvas.draw()

    
    def delete_data(self, event):
        os.remove(os.path.join(self.output_dir_path, str(self.current_folder), str(self.current_folder) + ".png"))
        os.remove(os.path.join(self.output_dir_path, str(self.current_folder),  str(self.current_folder) + ".json"))
        os.rmdir(os.path.join(self.output_dir_path, str(self.current_folder)))

        print(self.current_folder,  " deleted")


if __name__ == "__main__":
    
    output_dir_path = r"TuSimple_data_creation\test" # Path where manual labeller saved the data
    
    root = tk.Tk()
    app = LabelTool(root, output_dir_path=output_dir_path)
    root.mainloop()