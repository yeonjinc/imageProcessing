from tkinter import filedialog
import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk

# global variables
MARGIN = 10  # px
MAXDIM = 512

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
           WIDTH//2, HEIGHT+(2*MARGIN), 
           font="Tahoma 16",text="Original Photo")
        self.canvas1.create_text(
           WIDTH//2, HEIGHT+(2*MARGIN), 
           font="Tahoma 16",text="Changing Photo")
        
        
        
        ''' Feature Display Code'''
        # Create a FRAME that can fit the features
        self.frame2 = tk.Frame(self.window, width=self.width, height=self.height)
        self.frame2.pack(expand=1, fill=tk.X)
        
        # GUI Decription Text
        self.label_og = tk.Label(self.frame2, text="How-to-use GUI", font="Tahoma 20 bold")
        self.label_og.pack(anchor=tk.W)
        self.label_og = tk.Label(
            self.frame2, font="Tahoma 16", justify=tk.LEFT,
            text="This is Jenny Cho's GUI for E E 440 final project.\nUse the scale on the right to change your image however you want! \nHave fun :-)")
        self.label_og.pack(anchor=tk.W)
        
        # Create a CANVAS for buttons
        self.canvas_but = tk.Canvas(self.frame2, bd=2, bg="floral white")
        self.canvas_but.pack(side=tk.LEFT, anchor=tk.W)
        
        # Create many empty CANVASes
        for i in range(2):
            self.canvas = tk.Canvas(self.frame2, bg="floral white")
            self.canvas.pack(side=tk.LEFT)
        
        # Create a CANVAS for additional features
        self.canvas_feat = tk.Canvas(self.frame2, bd=2)
        self.canvas_feat.pack(side=tk.LEFT, anchor=tk.E)
        
        ''' Filter Feature Related Code'''
        # Create a BUTTON that loads image
        self.btn_load=tk.Button(
            self.canvas_but, text="Load", font="Tahoma 10 bold", command=self.load)
        self.btn_load.config(height=3, width=6)
        self.btn_load.pack(anchor=tk.NW)
  
        # Create a SCALE that lets the user blur the image
        self.scl_blur=tk.Scale(
            self.canvas_feat, from_=1, to=50, orient=tk.HORIZONTAL, 
            command = self.blur_image, sliderlength=50, 
            label="Blur", font="Tahoma 12")
        self.scl_blur.pack(anchor=tk.SE)
        
        # Create a SCALE that lets the user remove blemishes
        self.scl_blmsh=tk.Scale(
            self.canvas_feat, from_=0, to=10, orient=tk.HORIZONTAL, 
            command = self.decBlemish_image, sliderlength=50, 
            label="Blemish", font="Tahoma 12")
        self.scl_blmsh.pack(anchor=tk.SE)
        
        # Create a BUTTON that resets the image
        self.btn_reset=tk.Button(
            self.canvas_but, text="Reset", width=10, 
            command=self.reset, activeforeground="red3")
        self.btn_reset.pack(anchor=tk.W)
        
        # cartoon
        
        # Create a SCALE that lets the user mosaic the image
        self.scl_mosc=tk.Scale(
            self.canvas_feat, from_=1, to=40, orient=tk.HORIZONTAL, 
            command = self.mosaic_image, sliderlength=50, 
            label="Mosaic", font="Tahoma 12")
        self.scl_mosc.pack(anchor=tk.SE)

        self.window.mainloop()
        
        
        
    ''' Callback Functions'''
    # Callback for the "Blur" Scale
    def blur_image(self, k):
        k = self.scl_blur.get()
        self.NEWcv_img = cv2.blur(self.cv_img, (k, k))
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.NEWcv_img))
        self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo, anchor=tk.CENTER)
        
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
                    if (i*k) < img.shape[0] | (j*k) < img.shape[1]:
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
    
    # Callback for the "Load" Button
    def load(self):
        image_path = filedialog.askopenfilename()
        if (image_path):
            self.cv_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            self.height, self.width, no_channels = self.cv_img.shape
            if (self.height > MAXDIM) | (self.width > MAXDIM):
                # resize image to have max dim be 512 
                maxval = max(self.height, self.width)
                new_H = int(self.height/maxval)*MAXDIM
                new_W = int(self.width/maxval)*MAXDIM
                self.cv_img = cv2.resize(self.cv_img, (new_H,new_W))
            self.photoOG = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.cv_img))
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.cv_img))
            self.canvas0.create_image(MAXDIM//2, MAXDIM//2, image=self.photoOG)
            self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo)

    # Callback for the "Reset" Button
    def reset(self):
        # Reset Scales
        self.scl_blur.set(1)   # blur
        self.scl_blmsh.set(0)  # blemish
        self.scl_mosc.set(1)   # mosaic
        
        # Original Image
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
        self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo, anchor=tk.CENTER)  



# Create a window and pass it to the Application object
# App(tk.Toplevel(), "Tkinter and OpenCV")
App(tk.Tk(), "GUI Window", "jenny_resistor.bmp")
# App(tk.Tk(), "GUI Window")