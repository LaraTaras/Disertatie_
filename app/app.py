from tkinter import *
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.figure import Figure
from numpy import arange, sin, pi

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class App(tk.Tk):

    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):
        # __init__ function for class Tk
        tk.Tk.__init__(self, *args, **kwargs)

        # creating a container
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # initializing frames to an empty array
        self.frames = {}

        # iterating through a tuple consisting
        # of the different page layouts
        for F in (StartPage, PageOne, PageTwo):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        # Show a frame for the given page name
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="gray")
        self.controller = controller

        self.controller.title("Medical Application for Thyroid Nodule")
        # Designate Height and Width of the app
        app_height = 600
        app_width = 800

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        x = (screen_width / 2) - (app_width/2)
        y = (screen_height/2) - (app_height/2)

        self.controller.geometry(f'{app_width}x{app_height}+{int(x)}+{int(y)}')

        self.controller.resizable(width=False, height=False)
        self.controller.iconphoto(False, tk.PhotoImage(
                                                file='/Users/taras_lara/PycharmProjects/UNet/medical-symbol.png'))

        space_label = Label(self, bg="gray")
        space_label.pack(side=TOP, fill="x", pady=20)

        image_photo = tk.PhotoImage(file='medical-symbol.png')
        image_label = tk.Label(self, image=image_photo, bg="gray")
        image_label.pack()
        image_label.image = image_photo

        welcome_label = Label(self, text="WELCOME!", font=("Arial", 30, "bold"), width=10,
                              bg="gray",foreground="white", justify=LEFT)
        welcome_label.pack(side=TOP, fill="x", pady=5)

        option_label = Label(self,  text="Select an option to start.", font=("Arial", 20), width=30,
                             bg="gray",foreground="white")
        option_label.pack(side=TOP, fill="x", pady=5)

        space_label1 = Label(self,  height=2, width=5, bg="gray")
        space_label1.pack(side=TOP, fill="x", pady=80)

        mid_frame = tk.Frame(self, borderwidth=0, bg="gray")
        mid_frame.pack(fill='x')

        classification_btn = Button(mid_frame, text="Classification", font=("Arial", 20), bg="white", width=10, height=2,
                                    border="0", command=lambda: controller.show_frame("PageOne"))
        classification_btn.pack(side=LEFT, padx=80)

        segmentation_btn = Button(mid_frame, text="Segmentation", font=("Arial", 20), bg="white", width=10, height=2,
                                  border="0", command=lambda: controller.show_frame("PageTwo"))
        segmentation_btn.pack(side=RIGHT,padx=80)

        bottom_frame = tk.Frame(self, relief='raised', borderwidth=0, bg="gray33")
        bottom_frame.pack(fill='x', side=BOTTOM)

        space_label2 = tk.Label(bottom_frame, text='', font=("Arial", 30), bg="gray33")
        space_label2.pack()


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="gray")
        self.controller = controller
        img = 0
        pathtext = StringVar()
        IMG_WIDTH = 227
        IMG_HEIGHT = 227
        IMG_CHANNELS = 3

        def browsefunction():
            filepath = filedialog.askopenfilename()
            pathtext.set(filepath)
            image_display()

        def image_display():
            img = Image.open('%s' % str(pathtext.get()))
            resize_img = img.resize((300, 250))
            image = ImageTk.PhotoImage(resize_img)
            canvas.configure(image=image, width=0, height=0)
            canvas.image = image

        def get_prediction():
            X_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
            model = tf.keras.models.load_model('model1_alexnet_simple_aug.h5')
            ti_rads = tf.keras.models.load_model('model_alexnet_tirads_aug.h5')
            img = imread(str(pathtext.get()))[:, :, :IMG_CHANNELS]
            img = resize(img, (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), mode='constant', preserve_range=True)
            X_test[0] = img
            predictions_simple = model.predict(X_test, verbose=1)
            predictions_tirads = model.predict(X_test, verbose=1)

        space_label2 = Label(self, bg="gray")
        space_label2.pack(side=TOP, fill="x", pady=2)

        back_frame = tk.Frame(self, relief='raised', borderwidth=0, bg="gray")
        back_frame.pack(fill='x', side=TOP)

        go_back1 = tk.Button(back_frame, text="Go Back", font=("Arial", 15), bg="white", width=10, height=2, border="0",
                             command=lambda: controller.show_frame("StartPage"))
        go_back1.pack(side=LEFT, padx=10)

        classification_label = Label(self, text="Classification", font=("Arial", 30, "bold"), width=10,
                              bg="gray",foreground="White", justify=LEFT)
        classification_label.pack(side=TOP, fill="x", pady=10)

        mid_frame1 = tk.Frame(self, relief='raised', borderwidth=0, bg="gray")
        mid_frame1.pack(fill='x')

        upload_button = tk.Button(mid_frame1, text="Upload the image", font=("Arial", 15), bg="white", width=15,
                                  height=2, border="0", command=browsefunction)
        upload_button.pack(side=LEFT, padx=20, pady=15)

        space_label3 = Label(mid_frame1, bg="gray")
        space_label3.pack(side=LEFT, fill="x", padx=30)

        path_label = Label(mid_frame1, text="Path:", font=("Arial", 15, "bold"), width=5,
                           bg="gray", foreground="White")
        path_label.pack(side=LEFT, fill='x')

        path_entry = Entry(mid_frame1, textvariable=pathtext, font=("Arial", 15, "bold"), width=45,
                           bg="white", foreground="black", state="disabled")
        path_entry.pack(side=LEFT, padx=5)

        predict_button = tk.Button(self, text="Get Prediction", font=("Arial", 20), bg="white", width=10,
                                   height=2, border="0")
        predict_button.pack(pady=10)

        mid_frame2 = tk.Frame(self,  relief='raised', bg="gray")
        mid_frame2.pack()

        canvas = tk.Label(mid_frame2, width=33, height=15, bg="gray33")
        canvas.grid(row=0, column=0, rowspan=3)

        space_label4 = Label(mid_frame2, bg="gray", width=10)
        space_label4.grid(row=0, column=1, rowspan=3)

        tirads_label = Label(mid_frame2, text="TI-RADS:", font=("Arial", 15, "bold"),
                             bg="gray", foreground="White")
        tirads_label.grid(row=0, column=2)

        tirads_entry = Entry(mid_frame2, font=("Arial", 15, "bold"), width=20,
                           bg="white", foreground="black",state="disabled")
        tirads_entry.grid(row=0, column=3)

        type_label = Label(mid_frame2, text="TYPE:", font=("Arial", 15, "bold"),
                           bg="gray", foreground="White")
        type_label.grid(row=1, column=2)

        type_entry = Entry(mid_frame2, font=("Arial", 15, "bold"), width=20,
                           bg="white", foreground="black", state="disabled")
        type_entry.grid(row=1, column=3)

        trust_label = Label(mid_frame2, text="Trust:", font=("Arial", 15, "bold"),
                            bg="gray", foreground="White")
        trust_label.grid(row=2, column=2)
        # trust_label.pack(side=RIGHT)
        trust_entry = Entry(mid_frame2, font=("Arial", 15, "bold"), width=20,
                            bg="white", foreground="black", state="disabled")
        trust_entry.grid(row=2, column=3)

        bottom_frame = tk.Frame(self, relief='raised', borderwidth=0, bg="gray33")
        bottom_frame.pack(fill='x', side=BOTTOM)

        space_label5 = tk.Label(bottom_frame, text='', font=("Arial", 30), bg="gray33")
        space_label5.pack()


# third window frame page2
class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="gray")
        self.controller = controller

        pathtext = StringVar()
        IMG_WIDTH = 256
        IMG_HEIGHT = 256
        IMG_CHANNELS = 3
        trust = 0

        def browsefunction():
            filepath = filedialog.askopenfilename()
            pathtext.set(filepath)
            image_display()

        def image_display():
            img = Image.open('%s' % str(pathtext.get()))
            resize_img = img.resize((300, 250))
            image = ImageTk.PhotoImage(resize_img)
            canvas.configure(image=image, width=0, height=0)
            canvas.image = image

        def get_prediction():
            X_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
            model = tf.keras.models.load_model('model_FCN_ResNet50.h5')
            img = imread(str(pathtext.get()))[:, :, :IMG_CHANNELS]
            img = resize(img, (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), mode='constant', preserve_range=True)
            X_test[0] = img
            predictions = model.predict(X_test, verbose=1)
            preds_test_t = (predictions > 0.5).astype(np.uint8)
            im = plt.imshow(np.squeeze(preds_test_t[0]))
            f = Figure(figsize=(2.6, 2.6), dpi=100)
            a = f.add_subplot(111)
            t = arange(0.0, 3.0, 0.01)
            s = sin(2 * pi * t)
            a.imshow(np.squeeze(preds_test_t[0]), aspect='equal')
            a.axis('off')
            a.set_position([0, 0, 1, 1])
            canvas2 = FigureCanvasTkAgg(f, mid_frame2)
            canvas2.draw()
            canvas2.get_tk_widget().grid(row=0, column=3, rowspan=4)
            space_label6.configure(height=0, width=0)

        space_label2 = Label(self, bg="gray")
        space_label2.pack(side=TOP, fill="x", pady=2)

        back_frame = tk.Frame(self, relief='raised', borderwidth=0, bg="gray")
        back_frame.pack(fill='x', side=TOP)

        go_back1 = tk.Button(back_frame, text="Go Back", font=("Arial", 15), bg="white", width=10, height=2, border="0",
                             command=lambda: controller.show_frame("StartPage"))
        go_back1.pack(side=LEFT, padx=10)

        segmentation_label = Label(self, text="Segmentation", font=("Arial", 30, "bold"), width=10,
                                   bg="gray", foreground="White", justify=LEFT)
        segmentation_label.pack(side=TOP, fill="x", pady=5)

        mid_frame1 = tk.Frame(self, relief='raised', borderwidth=0, bg="gray")
        mid_frame1.pack(fill='x')

        upload_button = tk.Button(mid_frame1, text="Upload the image", font=("Arial", 15), bg="white", width=15,
                                  height=2, border="0", command=browsefunction)
        upload_button.pack(side=LEFT, padx=20, pady=15)
        path_label = Label(mid_frame1, text="Path:", font=("Arial", 15, "bold"), width=15,
                           bg="gray", foreground="White")
        path_label.pack(side=LEFT, fill='x', pady=10)

        path_entry = Entry(mid_frame1, textvariable=pathtext, font=("Arial", 15, "bold"), width=45,
                           bg="white", foreground="black", state="disabled")
        path_entry.pack(side=LEFT, padx=5)
        predict_button = tk.Button(self, text="Get Prediction", font=("Arial", 20), bg="white", width=10,
                                   height=2, border="0", command=get_prediction)
        predict_button.pack(pady=10)

        mid_frame2 = tk.Frame(self, relief='raised', borderwidth=0, bg="gray")
        mid_frame2.pack(fill='x')

        space_label4 = tk.Label(mid_frame2, width=1, bg="gray")
        space_label4.grid(row=0, column=0, rowspan=4)

        canvas = tk.Label(mid_frame2, borderwidth=3, width=33, height=15, bg="gray33")
        canvas.grid(row=0, column=1, rowspan=4)

        space_label5 = tk.Label(mid_frame2, width=1, bg="gray")
        space_label5.grid(row=0, column=4, rowspan=3)

        image_photo = tk.PhotoImage(file='right-arrow.png')
        image_label = tk.Label(mid_frame2, image=image_photo, bg="gray")
        image_label.image = image_photo
        image_label.grid(row=0, column=2, rowspan=4)

        space_label6 = tk.Label(mid_frame2, width=33, height=15, bg="gray33")
        space_label6.grid(row=0, column=3, rowspan=4)

        trust_label = Label(mid_frame2, text="Trust:", font=("Arial", 15, "bold"),
                            bg="gray", foreground="White")
        trust_label.grid(row=0, column=5)

        trust_entry = Entry(mid_frame2, font=("Arial", 15, "bold"), width=10,
                            bg="white", foreground="black", state="disabled")
        trust_entry.grid(row=1, column=5)

        iou_label = Label(mid_frame2, text="IOU:", font=("Arial", 15, "bold"),
                          bg="gray", foreground="White")
        iou_label.grid(row=2, column=5)

        iou_entry = Entry(mid_frame2, font=("Arial", 15, "bold"), width=10,
                          bg="white", foreground="black", state="disabled")
        iou_entry.grid(row=3, column=5)

        space_label7 = tk.Label(mid_frame2, width=10, bg="gray")
        space_label7.grid(row=4, column=5)

        bottom_frame = tk.Frame(self, relief='raised', borderwidth=0, bg="gray33")
        bottom_frame.pack(fill='x', side=BOTTOM)

        color_label6 = tk.Label(bottom_frame, text='', font=("Arial", 30), bg="gray33")
        color_label6.pack()


# Driver Code
app = App()
app.mainloop()
