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
import pickle
import tkinter.messagebox


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
                                                file='medical-symbol.png'))

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
        trust_tirads = StringVar()
        noduletype = StringVar()
        trust = StringVar()
        tiradstext = StringVar()
        pathtext = StringVar()
        IMG_WIDTH_tirads = 227
        IMG_HEIGHT_tirads = 227
        IMG_CHANNELS_tirads = 3

        IMG_WIDTH_simple = 150
        IMG_HEIGHT_simple = 150
        IMG_CHANNELS_simple = 3

        IMG_WIDTH_us = 224
        IMG_HEIGHT_us = 224
        IMG_CHANNELS_us = 3

        def browse_function():
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
            x_test_tirads = np.zeros((1, IMG_HEIGHT_tirads, IMG_WIDTH_tirads, IMG_CHANNELS_tirads), dtype=np.uint8)
            x_test_simple = np.zeros((1, IMG_HEIGHT_simple, IMG_WIDTH_simple, IMG_CHANNELS_simple), dtype=np.uint8)
            x_test_us = np.zeros((1, IMG_HEIGHT_us, IMG_WIDTH_us, IMG_CHANNELS_us), dtype=np.uint8)

            us = tf.keras.models.load_model('model_us_alexnet.h5')

            simple = tf.keras.models.load_model('model_inception_binary_bun.h5')
            with open('inceptionv3_simple.pkl', 'rb') as f:
                simple_history = pickle.load(f)

            ti_rads = tf.keras.models.load_model('model_alexnet_tirads_bun.h5')
            with open('alexnet_tirads.pkl', 'rb') as f:
                ti_rads_history = pickle.load(f)

            acc_simple = simple_history['accuracy']
            acc_simple = np.mean(acc_simple)
            acc_ti_rads = ti_rads_history['accuracy']
            acc_ti_rads = np.mean(acc_ti_rads)

            #metrics = np.mean(np.array([acc_simple, acc_ti_rads]), axis=0)

            img_tirads = imread(str(pathtext.get()))[:, :, :IMG_CHANNELS_tirads]
            img_tirads = resize(img_tirads, (IMG_WIDTH_tirads, IMG_HEIGHT_tirads, IMG_CHANNELS_tirads), mode='constant', preserve_range=True)
            x_test_tirads[0] = img_tirads

            img_simple = imread(str(pathtext.get()))[:, :, :IMG_CHANNELS_simple]
            img_simple = resize(img_simple, (IMG_WIDTH_simple, IMG_HEIGHT_simple, IMG_CHANNELS_simple), mode='constant', preserve_range=True)
            x_test_simple[0] = img_simple

            img_us = imread(str(pathtext.get()))[:, :, :IMG_CHANNELS_us]
            img_us = resize(img_us, (IMG_WIDTH_us, IMG_HEIGHT_us, IMG_CHANNELS_us), mode='constant', preserve_range=True)
            x_test_us[0] = img_us

            predictions_us = us.predict(x_test_us, verbose=1)
            predicted_classes_us = np.argmax(predictions_us, axis=1)

            predictions_simple = simple.predict(x_test_simple, verbose=1)
            predicted_classes_simple = np.argmax(predictions_simple, axis=1)

            predictions_ti_rads = ti_rads.predict(x_test_tirads, verbose=1)
            predicted_classes_ti_rads = np.argmax(predictions_ti_rads, axis=1)

            if predicted_classes_us == 0:
                if predicted_classes_simple == 0:
                    noduletype.set("benign")
                else:
                    noduletype.set("malign")

                if predicted_classes_ti_rads == 0:
                    tiradstext.set("2")
                elif predicted_classes_ti_rads == 1:
                    tiradstext.set("3")
                elif predicted_classes_ti_rads == 2:
                    tiradstext.set("4a")
                elif predicted_classes_ti_rads == 3:
                    tiradstext.set("4b")
                elif predicted_classes_ti_rads == 4:
                    tiradstext.set("4c")
                elif predicted_classes_ti_rads == 5:
                    tiradstext.set("5")

                acc_ti_rads = acc_ti_rads * 100
                trust_tirads.set(str("%.2f" % acc_ti_rads) + "%")

                acc_simple = acc_simple * 100
                trust.set(str("%.2f" % acc_simple) + "%")
            else:
                tkinter.messagebox.showinfo('Error Message', 'Error: Please select a US image!')
                trust_tirads.set("NULL")
                trust.set("NULL")
                tiradstext.set("NULL")
                noduletype.set("NULL")


            # noduletype.set(str(predicted_classes_simple))
            # metrics = simple.evaluate(X_test, predictions_simple)


        space_label2 = Label(self, bg="gray")
        space_label2.pack(side=TOP, fill="x", pady=2)

        back_frame = tk.Frame(self, relief='raised', borderwidth=0, bg="gray")
        back_frame.pack(fill='x', side=TOP)

        go_back1 = tk.Button(back_frame, text="Go Back", font=("Arial", 15), bg="white", width=10, height=2, border="0",
                             command=lambda: controller.show_frame("StartPage"))
        go_back1.pack(side=LEFT, padx=10)

        classification_label = Label(self, text="Classification", font=("Arial", 30, "bold"), width=10,
                              bg="gray", foreground="White", justify=LEFT)
        classification_label.pack(side=TOP, fill="x", pady=10)

        mid_frame1 = tk.Frame(self, relief='raised', borderwidth=0, bg="gray")
        mid_frame1.pack(fill='x')

        upload_button = tk.Button(mid_frame1, text="Upload the image", font=("Arial", 15), bg="white", width=15,
                                  height=2, border="0", command=browse_function)
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
                                   height=2, border="0", command=get_prediction)
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

        tirads_entry = Entry(mid_frame2, font=("Arial", 15, "bold"), textvariable=tiradstext, width=20,
                             bg="white", foreground="black", state="disabled")
        tirads_entry.grid(row=0, column=3)

        trust_tirads_label = Label(mid_frame2, text="Trust TI-RADS:", font=("Arial", 15, "bold"),
                            bg="gray", foreground="White")
        trust_tirads_label.grid(row=1, column=2)
        # trust_label.pack(side=RIGHT)
        trust_tirads_entry = Entry(mid_frame2, font=("Arial", 15, "bold"), textvariable=trust_tirads, width=20,
                            bg="white", foreground="black", state="disabled")
        trust_tirads_entry.grid(row=1, column=3)

        type_label = Label(mid_frame2, text="TYPE:", font=("Arial", 15, "bold"),
                           bg="gray", foreground="White")
        type_label.grid(row=2, column=2)

        type_entry = Entry(mid_frame2, font=("Arial", 15, "bold"), textvariable=noduletype, width=20,
                           bg="white", foreground="black", state="disabled")
        type_entry.grid(row=2, column=3)

        trust_label = Label(mid_frame2, text="Trust TYPE:", font=("Arial", 15, "bold"),
                            bg="gray", foreground="White")
        trust_label.grid(row=3, column=2)
        # trust_label.pack(side=RIGHT)
        trust_entry = Entry(mid_frame2, font=("Arial", 15, "bold"), textvariable=trust, width=20,
                            bg="white", foreground="black", state="disabled")
        trust_entry.grid(row=3, column=3)

        bottom_frame = tk.Frame(self, relief='raised', borderwidth=0, bg="gray33")
        bottom_frame.pack(fill='x', side=BOTTOM)

        space_label5 = tk.Label(bottom_frame, text='', font=("Arial", 30), bg="gray33")
        space_label5.pack()


# third window frame page2
class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="gray")
        self.controller = controller

        iou = StringVar()
        trust = StringVar()
        pathtext = StringVar()
        IMG_WIDTH = 256
        IMG_HEIGHT = 256
        IMG_CHANNELS = 3

        height = 224
        width = 224

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
            x_test = np.zeros((1, height, width, IMG_CHANNELS), dtype=np.uint8)
            X_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

            us = tf.keras.models.load_model('model_us_alexnet.h5')
            model = tf.keras.models.load_model('model_FCN_ResNet50_bun3.h5')
            with open('fcn_resnet50.pkl', 'rb') as f:
                fcn_resnet_history = pickle.load(f)

            acc_fcn_resnet = fcn_resnet_history['accuracy']
            acc_fcn_resnet = np.mean(acc_fcn_resnet)

            img = imread(str(pathtext.get()))[:, :, :IMG_CHANNELS]
            img = resize(img, (width, height, IMG_CHANNELS), mode='constant', preserve_range=True)
            x_test[0] = img

            img = imread(str(pathtext.get()))[:, :, :IMG_CHANNELS]
            img = resize(img, (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), mode='constant', preserve_range=True)
            X_test[0] = img

            predictions_us = us.predict(x_test, verbose=1)
            predicted_classes_us = np.argmax(predictions_us, axis=1)

            predictions = model.predict(X_test, verbose=1)
            preds_test_t = (predictions > 0.5).astype(np.uint8)

            # im = plt.imshow(np.squeeze(preds_test_t[0]))
            f = Figure(figsize=(2.6, 2.6), dpi=100)
            a = f.add_subplot(111)
            # t = arange(0.0, 3.0, 0.01)
            # s = sin(2 * pi * t)

            a.imshow(np.squeeze(preds_test_t[0]), extent=[0,100,0,1], aspect='auto')
            a.axis('off')
            a.set_position([0, 0, 1, 1])
            canvas2 = FigureCanvasTkAgg(f, mid_frame2)
            if predicted_classes_us == 0:
                canvas2.draw()
                canvas2.get_tk_widget().grid(row=0, column=3, rowspan=4)

                space_label6.configure(height=0, width=0)

                acc_fcn_resnet = acc_fcn_resnet * 100
                trust.set(str("%.2f" % acc_fcn_resnet)+"%")
            else:
                tkinter.messagebox.showinfo('Error Message', 'Error: Please select a US image!')
                f.clear()
                a = f.add_subplot(111)
                a.clear()
                a.set_position([0, 0, 1, 1])
                canvas2 = FigureCanvasTkAgg(f, mid_frame2)
                canvas2.draw()
                canvas2.get_tk_widget().grid(row=0, column=3, rowspan=4)
                # canvas2.draw_idle()
                trust.set("NULL")

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

        trust_entry = Entry(mid_frame2, font=("Arial", 15, "bold"), textvariable=trust, width=10,
                            bg="white", foreground="black", state="disabled")
        trust_entry.grid(row=1, column=5)

        # iou_label = Label(mid_frame2, text="IOU:", font=("Arial", 15, "bold"),
        #                   bg="gray", foreground="White")
        # iou_label.grid(row=2, column=5)
        #
        # iou_entry = Entry(mid_frame2, font=("Arial", 15, "bold"), width=10,
        #                   bg="white", foreground="black", state="disabled")
        # iou_entry.grid(row=3, column=5)

        space_label7 = tk.Label(mid_frame2, width=10, bg="gray")
        space_label7.grid(row=4, column=5)

        bottom_frame = tk.Frame(self, relief='raised', borderwidth=0, bg="gray33")
        bottom_frame.pack(fill='x', side=BOTTOM)

        color_label6 = tk.Label(bottom_frame, text='', font=("Arial", 30), bg="gray33")
        color_label6.pack()


# Driver Code
app = App()
app.mainloop()