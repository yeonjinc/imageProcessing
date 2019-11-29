''' 
Written by Jenny Cho, ©JCHOLOL, 2019

Description
This program is a GUI built using tkinter package. It displays a set of images, 
    one original (left) and one changing. The changing image reflects the effects 
    being applied to the original image.
    Not implemented: The image effects should be able to build on top of each other 

Parameters
    window:       tk.Tk() or tk.Toplevel();  Tk() works best
    window_title: title of the window
    image_path:   image 
                  (compatible types: jpg, jpeg, png, bmp, tiff, pbm, pgm, ppm)

! DO NOT USE MY CODE WITHOUT GIVING ME CREDIT !
! YOU DONT KNOW HOW MANY mL OF TEARS I SHED !
! MAKING THE GUI !
! SCREW YOU ! 
(or thank you!)

If my code can save you some hours of depression...sure. 
    You can use it.
But you should hoot me an email at 
    aboynamedbob [at] gmail [dot] com and thank me ;)
'''
from tkinter import filedialog
import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np

# global variables
MARGIN = 10  # px
MAXDIM = 530
SLIDER_SIZE = 20
SV = 0  # show value of scale (1 or 0)

class App():
    def __init__(self, window, window_title, image_path="lena.bmp"):
        self.window = window
        self.window.title(window_title)
        
        # Load an image using OpenCV
        self.cv_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        
        # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
        self.height, self.width, no_channels = self.cv_img.shape
        
        ''' Image Display Related Code'''
        # Create a FRAME that can fit the images
        self.frame1 = tk.Frame(self.window, width=self.width, height=self.height)
        self.frame1.pack(fill=tk.BOTH)        
        
        # Create a FRAME for original image
        self.frame_original = tk.Frame(self.frame1, width=self.width, height=self.height)
        self.frame_original.pack(side=tk.LEFT)
        
        # Create a CANVAS for original image
        self.canvas0 = tk.Canvas(self.frame_original, width=MAXDIM, height=MAXDIM+(3*MARGIN))
        self.canvas0.pack()
        
        # Create a FRAME for changing image
        self.frame_new = tk.Frame(self.frame1, width=self.width, height=self.height)
        self.frame_new.pack(side=tk.LEFT)
        
        # Create a CANVAS for changing image
        self.canvas1 = tk.Canvas(self.frame_new, width=MAXDIM, height=MAXDIM+(3*MARGIN))
        self.canvas1.pack()

        # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
        self.photoOG = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.cv_img))
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.cv_img))
        
        # Add a PhotoImage to the Canvas (original)
        self.canvas0.create_image(MAXDIM//2, MAXDIM//2, image=self.photoOG)
        
        # Add a PhotoImage to the Canvas (changing effects)
        self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo, anchor=tk.CENTER)
        
        # Write labels for both images
        self.canvas0.create_text(
           MAXDIM//2, MAXDIM+(2*MARGIN), 
           font="Tahoma 16",text="Original Photo")
        self.canvas1.create_text(
           MAXDIM//2, MAXDIM+(2*MARGIN), 
           font="Tahoma 16",text="Changing Photo")
        
        
##############################################################################################
##################################   FEATURE DISPLAY   #######################################
##############################################################################################

        # Create a FRAME that can fit the features
        self.frame2 = tk.Frame(self.window, width=self.width, height=self.height//2)
        self.frame2.pack(expand=1, fill=tk.X)
        
        # GUI Decription Text
        self.label_og = tk.Label(self.frame2, text="How-to-use GUI", font="Tahoma 20 bold")
        self.label_og.pack(anchor=tk.W)
        self.label_og = tk.Label(
            self.frame2, font="Tahoma 16", justify=tk.LEFT,
            text="This is Jenny Cho's GUI for E E 440 final project. Use the scale on the right to change your image however you want! Have fun :-)")
        self.label_og.pack(anchor=tk.W)
        
        # Create a CANVAS for buttons
        self.canvas_but = tk.Canvas(self.frame2, bd=2)
        self.canvas_but.pack(side=tk.LEFT, anchor=tk.W)
        
        #%$@%$@%$@%$@_____SPACER (between buttons and scales)
        self.canvas = tk.Canvas(self.frame2, height=175, width=600)
        self.canvas.pack(side=tk.LEFT)
        
        # Create a CANVAS for additional features (1)
        #  blur & blemish & mosaic
        self.canvas_feat1 = tk.Canvas(self.frame2)
        self.canvas_feat1.pack(side=tk.LEFT, anchor=tk.E)
        
        ##%$@%$@%$@%$@_____SPACER (between scales)
        self.canvas = tk.Canvas(self.frame2, height=175, width=30)
        self.canvas.pack(side=tk.LEFT)
        
        # Create a CANVAS for additional features (2)
        #  gamma & brightness & contrast
        self.canvas_feat2 = tk.Canvas(self.frame2)
        self.canvas_feat2.pack(side=tk.LEFT, anchor=tk.E)
        
        ##%$@%$@%$@%$@_____SPACER (between scales)
        self.canvas = tk.Canvas(self.frame2, height=175, width=30)
        self.canvas.pack(side=tk.LEFT)
        
        # Create a CANVAS for additional features (3)
        #   sharpness & 
        self.canvas_feat3 = tk.Canvas(self.frame2, bd=2)
        self.canvas_feat3.pack(side=tk.LEFT, anchor=tk.E)
        
    
##############################################################################################
######################################   WIDGETS   ###########################################
##############################################################################################

        # Create a BUTTON that loads image
        self.btn_load = tk.Button(
            self.canvas_but, text="Load", font="Tahoma 10", command=self.load, 
            activeforeground="black")
        self.btn_load.config(height=3, width=9)
        self.btn_load.pack(anchor=tk.CENTER)
        
        #%$@%$@%$@%$@_____SPACER (between two buttons)
        self.canvas = tk.Canvas(self.canvas_but, height=100, width=50)
        self.canvas.pack()
  
        # Create a BUTTON that resets the image
        self.btn_reset = tk.Button(
            self.canvas_but, text="Reset", font="Tahoma 8",
            command=self.reset, activeforeground="red3")
        self.btn_reset.config(height=1, width=9)
        self.btn_reset.pack(anchor=tk.CENTER)
        
        
        # Create a SCALE that lets the user blur the image
        self.scl_blur = tk.Scale(
            self.canvas_feat1, from_=1, to=50, orient=tk.HORIZONTAL, showvalue=SV,
            command = self.blur_image, sliderlength=SLIDER_SIZE, 
            label="Blur", font="Tahoma 12")
        self.scl_blur.pack(anchor=tk.SE)
        
        # Create a SCALE that lets the user remove blemishes
        self.scl_blmsh = tk.Scale(
            self.canvas_feat1, from_=0, to=10, orient=tk.HORIZONTAL, showvalue=SV,
            command = self.decBlemish_image, sliderlength=SLIDER_SIZE, 
            label="Blemish", font="Tahoma 12")
        self.scl_blmsh.pack(anchor=tk.SE)
        
        # Create a SCALE that lets the user mosaic the image
        self.scl_mosc = tk.Scale(
            self.canvas_feat1, from_=1, to=40, orient=tk.HORIZONTAL, showvalue=SV,
            command = self.mosaic_image, sliderlength=SLIDER_SIZE, 
            label="Mosaic", font="Tahoma 12")
        self.scl_mosc.pack(anchor=tk.SE)
        
        
        # Create a SCALE that lets the user apply gamma correction
        self.scl_gamma = tk.Scale(
            self.canvas_feat2, from_=0.5, to=3.5, orient=tk.HORIZONTAL, showvalue=SV,
            command = self.gamma_image, sliderlength=SLIDER_SIZE, resolution=0.02,
            label="Gamma Correction", font="Tahoma 12")
        self.scl_gamma.set(1)
        self.scl_gamma.pack(anchor=tk.SE)
        
        # Create a SCALE that lets the user change brightness
        self.scl_bright = tk.Scale(
            self.canvas_feat2, from_=-25, to=25, orient=tk.HORIZONTAL, showvalue=SV,
            command = self.brightness_image, sliderlength=SLIDER_SIZE, 
            label="Brightness", font="Tahoma 12")
        self.scl_bright.pack(anchor=tk.SE)
        
        # Create a SCALE that lets the user change contrast
        self.scl_contrast = tk.Scale(
            self.canvas_feat2, from_=13, to=113, orient=tk.HORIZONTAL, showvalue=SV,
            command = self.contrast_image, sliderlength=SLIDER_SIZE, 
            label="Contrast", font="Tahoma 12")
        self.scl_contrast.set(63)
        self.scl_contrast.pack(anchor=tk.SE)
        
        # Create a SCALE that lets the user change sharpness
        self.scl_sharpen = tk.Scale(
            self.canvas_feat3, from_=0, to=1.5, orient=tk.HORIZONTAL, showvalue=SV,
            command = self.sharpen_image, sliderlength=SLIDER_SIZE, resolution=0.005,
            label="Sharpness", font="Tahoma 12")
        self.scl_sharpen.pack(anchor=tk.SE)
        
        # Create a SCALE that lets the user pencil sketch the image
        self.scl_sketch = tk.Scale(
            self.canvas_feat3, from_=2, to=30, orient=tk.HORIZONTAL, showvalue=SV,
            command = self.sketch_image, sliderlength=SLIDER_SIZE, resolution=2,
            label="Pencil Sketch", font="Tahoma 12")
        self.scl_sketch.pack(anchor=tk.SE)
    
    
        # Create a SCALE that lets the user vignette the image
        self.scl_vignette = tk.Scale(
            self.canvas_feat3, from_=0, to=100-1, orient=tk.HORIZONTAL, showvalue=SV,
            command = self.vignette_image, sliderlength=SLIDER_SIZE,
            label="Vignette", font="Tahoma 12")
        self.scl_vignette.pack(anchor=tk.SE)

        self.window.mainloop()


##############################################################################################
#################################  CALLBACK FUNCTIONS  #######################################
##############################################################################################

    '''#################################  LOAD  ###############################'''
    # Callback for the "Load" Button
    def load(self):
        image_path = filedialog.askopenfilename()
        if (image_path):
            self.cv_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            self.height, self.width, no_channels = self.cv_img.shape
            if (self.height > MAXDIM) or (self.width > MAXDIM):
                # resize image to have max dim be 512 
                maxval = max(self.height, self.width)
                scale = MAXDIM/maxval  # portion of original size (decimal)
                new_H = int(self.height * scale)
                new_W = int(self.width * scale)
                self.cv_img = cv2.resize(self.cv_img, (new_W,new_H))
            
            self.reset()
            
            self.photoOG = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.cv_img))
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.cv_img))
            self.canvas0.create_image(MAXDIM//2, MAXDIM//2, image=self.photoOG)
            self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo)


    '''#################################  RESET  ###############################'''
    # Callback for the "Reset" Button
    def reset(self):
        # Reset Scales
        self.scl_blur.set(1)      # blur
        self.scl_blmsh.set(0)     # blemish
        self.scl_mosc.set(1)      # mosaic
        
        self.scl_gamma.set(1)     # gamma
        self.scl_bright.set(0)    # brightness
        self.scl_contrast.set(63) # contrast
        
        self.scl_sharpen.set(0)   # sharpen
        self.scl_sketch.set(2)    # pencil sketch
        self.scl_vignette.set(0)  # vignette
        
        # Original Image
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
        self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo, anchor=tk.CENTER) 
        
        
    '''#################################  BLUR  ###############################'''
    # Callback for the "Blur" Scale 
    def blur_image(self, k):
        k = self.scl_blur.get()
        self.NEWcv_img = cv2.blur(self.cv_img, (k, k))
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.NEWcv_img))
        self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo, anchor=tk.CENTER)
        

    '''#################################  BLEMISH  ###############################'''
    # Callback for the "Blemish" Scale
    def decBlemish_image(self, k):
        k = self.scl_blmsh.get()
        # cancel out the effect
        if k == 0:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
            self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo, anchor=tk.CENTER)
        else:
            sigmaColor = 75
            sigmaSpace = 10
            self.NEWcv_img = cv2.bilateralFilter(self.cv_img, k*4, sigmaColor, sigmaSpace)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.NEWcv_img))
            self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo, anchor=tk.CENTER)
    
    
    '''################################  MOSAIC  ###############################'''
    # Mosaic Effect
    def mosaic_effect(self, k):
        img_mosc = np.zeros_like(self.cv_img)
        for ch in range(3):  # all three bgr channels
            img = self.cv_img[:,:,ch]

            # save a "small photo" for every "k"
            img_small = img[0::k, 0::k]
            h, w = img_small.shape

            # new image frame
            x, y, ignore = self.cv_img.shape
            img_ch = np.zeros((x,y))

            # fill picture with mosaic-ed pixels
            for i in range(h):
                for j in range(w):
                    if ((i*k) < img.shape[0]) or ((j*k) < img.shape[1]):
                        img_ch[(i*k):(i*k)+k, (j*k):(j*k)+k] = img_small[i][j]

            img_mosc[:,:,ch] = img_ch
        return img_mosc
                
    # Callback for the "Mosaic" Scale
    def mosaic_image(self, k):
        k = self.scl_mosc.get()
        if k == 1:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
            self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo, anchor=tk.CENTER)
        else:
            self.NEWcv_img = self.mosaic_effect(k)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.NEWcv_img))
            self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo, anchor=tk.CENTER)

            
    '''############################  GAMMA CORRECTION  ###############################'''
    # Gamma Correction / Power Law Transform
    #   build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    def gamma_effect(self, gamma=2):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)

        # apply gamma correction using the lookup table
        return cv2.LUT(self.cv_img, table)
    
    # Callback for the "Gamma Correction" Scale
    # reference  >> https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    def gamma_image(self, k):
        k = self.scl_gamma.get()
        self.NEWcv_img = self.gamma_effect(gamma=k)
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.NEWcv_img))
        self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo, anchor=tk.CENTER)
        
        
    '''################################  BRIGHTNESS  ###############################'''
    # Brightness Effect
    def brightness_effect(self, k):
        img_new = np.zeros_like(self.cv_img)
        if np.array_equal(self.cv_img[:,:,0],self.cv_img[:,:,1]):  # grayscale
            img_ch = (self.cv_img[:,:,0].astype(np.uint16))+(k*5)  # add 5*k value channel
            img_ch[img_ch < 0] = 0      # adjust wrap around pixel (bottom)
            img_ch[img_ch > 255] = 255  # adjust overflow (top)
            img_new[:,:,0] = img_ch
            img_new[:,:,1] = img_ch
            img_new[:,:,2] = img_ch
        else:
            for ch in range(3):  # all three bgr channels
                img_ch = (self.cv_img[:,:,ch].astype(np.uint16))+(k*5)  # add 5*k value channel
                img_ch[img_ch < 0] = 0      # adjust wrap around pixel (bottom)
                img_ch[img_ch > 255] = 255  # adjust overflow (top)
                img_new[:,:,ch] = img_ch
        return img_new.astype(np.uint8)
    
    # Callback for the "Brightness" Scale
    def brightness_image(self, k):
        k = self.scl_bright.get()
        self.NEWcv_img = self.brightness_effect(k)
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.NEWcv_img))
        self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo, anchor=tk.CENTER)
        
 
    '''###############################  CONTRAST  ###############################'''
    # Change contrast of image
    #   scale [0, 127] starts at 27   (r1=s1 = 100; r2=s2 = 154)
    #   scale [0, 127] starts at 63   (r1=s1 =  64; r2=s2 = 190)
    def contrast_effect(self, k):
        r1, r2 = 64, 190  # fixed input image values
        s1, s2 = 127-k, 127+k

        img_new = self.cv_img.astype(np.uint16)
        
        slope1 = s1 / r1
        mask1 = (img_new >= 0) & (img_new <= r1)
        img_new[mask1] = slope1*img_new[mask1]
        
        slope2 = (s2-s1) / (r2-r1)
        mask2 = (img_new > r1) & (img_new <= r2)
        img_new[mask2] = slope2*(img_new[mask2]-r1) + s1
        
        slope3 = (255-s2) / (255-r2)
        mask3 = (img_new > r2) & (img_new < 256)
        img_new[mask3] = slope3*(img_new[mask3]-r2) + s2

        return img_new.astype(np.uint8)
    
    # Callback for the "Contrast" Scale
    def contrast_image(self, k):
        k = self.scl_contrast.get()
        self.NEWcv_img = self.contrast_effect(k)
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.NEWcv_img))
        self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo, anchor=tk.CENTER)
        
    
    '''################################  SHARPEN  ###############################'''
    def sharpen_image(self, k):
        k = self.scl_sharpen.get()
        hsv = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2HSV)
        img_v = hsv[:,:,2]  # value channel

        mask = np.array([[-k,    -k, -k],
                         [-k, 1+8*k, -k],
                         [-k,    -k, -k]]) # og + k*mask
        hsv[:,:,2] = cv2.filter2D(img_v, 64, mask)
        self.NEWcv_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.NEWcv_img))
        self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo, anchor=tk.CENTER)

    
    '''############################  PENCIL SKETCH  ###########################'''
    def dodge(self, top, bottom):
        return cv2.divide(top, 255-bottom, scale=256)
    
    # k = [4, 30] --> [3, 29]
    # k == 2: "reset"
    def sketch_image(self, k):
        # Invert it original image
        # Blur the inverted image
        k = self.scl_sketch.get()-1  # -1 to make k odd
        if k == 1:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
            self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo, anchor=tk.CENTER)
        else:
            self.gray_img = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2GRAY)
            self.inv_img = 255-self.gray_img
            self.blur_img = cv2.GaussianBlur(self.inv_img, (k, k), sigmaX=k)
            # Dodge blend the blurred and grayscale image.
            self.NEWcv_img = self.dodge(self.gray_img, self.blur_img)
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.NEWcv_img))
            self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo, anchor=tk.CENTER)
    
    
    '''############################  VIGNETTE  ###########################'''
    # k = [0, 100-1]
    def vignette_image(self, k):
        k = self.scl_vignette.get()
        row, col, ignore = self.cv_img.shape
        # mask (oval)
        black = np.zeros_like(self.cv_img)
        center_coordinates = (col//2, row//2)
        radius = (max(row,col)//2) + (100-k)
        color = (255, 255, 255)  # white
        thickness = -1
        mask = cv2.circle(black, center_coordinates, radius, color, thickness) 
        mask = cv2.blur(mask, (200, 200)).astype(float)/255
        img = np.copy(self.cv_img).astype(float)/255
        self.NEWcv_img = (self.cv_img * mask).astype(np.uint8)
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.NEWcv_img))
        self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo, anchor=tk.CENTER)

# Create a window and pass it to the Application object
# App(tk.Toplevel(), "Tkinter and OpenCV")
App(tk.Tk(), "Jenny's E E 440 Final Project")