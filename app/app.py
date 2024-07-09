from tkinter import *
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import matplotlib
from matplotlib.figure import Figure
import pickle
import tkinter.messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

matplotlib.use('TkAgg')

# Crearea clasei principale App
class App(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        # Crearea unui container
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # Inițializarea vectorului de pagini
        self.frames = {}

        # Se iterează prin vectorul de pagini
        for F in (StartPage, PageOne, PageTwo):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        # Se arată frame-ul pentru fiecare pagină
        frame = self.frames[page_name]
        frame.tkraise()


# se va crea pagina principală
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="gray")
        self.controller = controller

        self.controller.title("Medical Application for Thyroid Nodule")
        # Înălțimea și lungimea aplicației
        app_height = 600
        app_width = 800

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        x = (screen_width/2) - (app_width/2)
        y = (screen_height/2) - (app_height/2)

        # Se va redimensioma și se va poziționa la mijlocul ecranului.
        self.controller.geometry(f'{app_width}x{app_height}+{int(x)}+{int(y)}')
        self.controller.resizable(width=False, height=False)
        # Se va adăuga un icon
        self.controller.iconphoto(False, tk.PhotoImage(file='medical-symbol.png'))

        # se introduc spații
        space_label = Label(self, bg="gray")
        space_label.pack(side=TOP, fill="x", pady=20)

        #Se introduce imaginea pe fundal
        image_photo = tk.PhotoImage(file='medical-symbol.png')
        image_label = tk.Label(self, image=image_photo, bg="gray")
        image_label.pack()
        image_label.image = image_photo

        # se adaugă mesajul de intampinare
        welcome_label = Label(self, text="WELCOME!", font=("Arial", 30, "bold"), width=10,
                              bg="gray", foreground="white", justify=LEFT)
        welcome_label.pack(side=TOP, fill="x", pady=5)

        # se adaugă eticheta de opține
        option_label = Label(self,  text="Select an option to start.", font=("Arial", 20), width=30,
                             bg="gray", foreground="white")
        option_label.pack(side=TOP, fill="x", pady=5)

        space_label1 = Label(self,  height=2, width=5, bg="gray")
        space_label1.pack(side=TOP, fill="x", pady=80)

        # se împarte pagina în mai multe părți/ frame-uri
        mid_frame = tk.Frame(self, borderwidth=0, bg="gray")
        mid_frame.pack(fill='x')

        # se adaugă butoanele de clasificare și segmentare
        classification_btn = Button(mid_frame, text="Classification", font=("Arial", 20), bg="white", width=10,
                                    height=2, border="0", command=lambda: controller.show_frame("PageOne"))
        classification_btn.pack(side=LEFT, padx=80)

        segmentation_btn = Button(mid_frame, text="Segmentation", font=("Arial", 20), bg="white", width=10, height=2,
                                  border="0", command=lambda: controller.show_frame("PageTwo"))
        segmentation_btn.pack(side=RIGHT, padx=80)

        # se adaugă partea de jos având o culoare gri închis
        bottom_frame = tk.Frame(self, relief='raised', borderwidth=0, bg="gray33")
        bottom_frame.pack(fill='x', side=BOTTOM)

        space_label2 = tk.Label(bottom_frame, text='', font=("Arial", 30), bg="gray33")
        space_label2.pack()


#Se creează pagina de clasificare
class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="gray")
        self.controller = controller

        # se adaugă variabilele necesare
        trust_ti_rads = StringVar()
        nodule_type = StringVar()
        trust = StringVar()
        ti_rads_text = StringVar()
        path_text = StringVar()

        # dimensiunea modelului de clasificare multi-clasă
        img_width_ti_rads = 227
        img_height_ti_rads = 227
        img_channels_ti_rads = 3

        # dimensiunea modelului de clasificare simplă
        img_width_simple = 150
        img_height_simple = 150
        img_channels_simple = 3

        # dimensiunea modelului de diferențiere
        img_width_us = 224
        img_height_us = 224
        img_channels_us = 3

        # se creează funcția de căutare
        def browse_function():
            filepath = filedialog.askopenfilename()
            path_text.set(filepath)
            image_display()

        # se afișează imaginea
        def image_display():
            img = Image.open('%s' % str(path_text.get()))
            resize_img = img.resize((300, 250))
            image = ImageTk.PhotoImage(resize_img)
            canvas.configure(image=image, width=0, height=0)
            canvas.image = image

        # se creează o funcție de predicție
        def get_prediction():
            # se creează variabilele
            x_test_ti_rads = np.zeros((1, img_height_ti_rads, img_width_ti_rads, img_channels_ti_rads), dtype=np.uint8)
            x_test_simple = np.zeros((1, img_height_simple, img_width_simple, img_channels_simple), dtype=np.uint8)
            x_test_us = np.zeros((1, img_height_us, img_width_us, img_channels_us), dtype=np.uint8)

            # se adaugă modelele
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

            # se citește imaginea aleasă de utilizator
            img_ti_rads = imread(str(path_text.get()))[:, :, :img_channels_ti_rads]
            img_ti_rads = resize(img_ti_rads, (img_width_ti_rads, img_height_ti_rads, img_channels_ti_rads),
                                 mode='constant', preserve_range=True)
            x_test_ti_rads[0] = img_ti_rads

            img_simple = imread(str(path_text.get()))[:, :, :img_channels_simple]
            img_simple = resize(img_simple, (img_width_simple, img_height_simple, img_channels_simple),
                                mode='constant', preserve_range=True)
            x_test_simple[0] = img_simple

            img_us = imread(str(path_text.get()))[:, :, :img_channels_us]
            img_us = resize(img_us, (img_width_us, img_height_us, img_channels_us),
                            mode='constant', preserve_range=True)
            x_test_us[0] = img_us

            # se generează predicțiile
            predictions_us = us.predict(x_test_us, verbose=1)
            predicted_classes_us = np.argmax(predictions_us, axis=1)

            predictions_simple = simple.predict(x_test_simple, verbose=1)
            predicted_classes_simple = np.argmax(predictions_simple, axis=1)

            predictions_ti_rads = ti_rads.predict(x_test_ti_rads, verbose=1)
            predicted_classes_ti_rads = np.argmax(predictions_ti_rads, axis=1)

            # se afișează răspunsul
            if predicted_classes_us == 0:
                if predicted_classes_simple == 0:
                    nodule_type.set("benign")
                else:
                    nodule_type.set("malign")

                if predicted_classes_ti_rads == 0:
                    ti_rads_text.set("2")
                elif predicted_classes_ti_rads == 1:
                    ti_rads_text.set("3")
                elif predicted_classes_ti_rads == 2:
                    ti_rads_text.set("4a")
                elif predicted_classes_ti_rads == 3:
                    ti_rads_text.set("4b")
                elif predicted_classes_ti_rads == 4:
                    ti_rads_text.set("4c")
                elif predicted_classes_ti_rads == 5:
                    ti_rads_text.set("5")

                acc_ti_rads = acc_ti_rads * 100
                trust_ti_rads.set(str("%.2f" % acc_ti_rads) + "%")

                acc_simple = acc_simple * 100
                trust.set(str("%.2f" % acc_simple) + "%")
                # dacă nu este o imagine medicală se generează o fereastră de eroare
            else:
                tkinter.messagebox.showinfo('Error Message', 'Error: Please select a US image!')
                trust_ti_rads.set("NULL")
                trust.set("NULL")
                ti_rads_text.set("NULL")
                nodule_type.set("NULL")

        # se adaugă spații
        space_label2 = Label(self, bg="gray")
        space_label2.pack(side=TOP, fill="x", pady=2)

        back_frame = tk.Frame(self, relief='raised', borderwidth=0, bg="gray")
        back_frame.pack(fill='x', side=TOP)

        # se adaugă butonul de mers inapoi
        go_back1 = tk.Button(back_frame, text="Go Back", font=("Arial", 15), bg="white", width=10, height=2, border="0",
                             command=lambda: controller.show_frame("StartPage"))
        go_back1.pack(side=LEFT, padx=10)

        # se adaugă eticheta de clasificare a titlului
        classification_label = Label(self, text="Classification", font=("Arial", 30, "bold"), width=10,
                                     bg="gray", foreground="White", justify=LEFT)
        classification_label.pack(side=TOP, fill="x", pady=10)

        mid_frame1 = tk.Frame(self, relief='raised', borderwidth=0, bg="gray")
        mid_frame1.pack(fill='x')

        # se adaugă butonul de încărcare a imaginii
        upload_button = tk.Button(mid_frame1, text="Upload the image", font=("Arial", 15), bg="white", width=15,
                                  height=2, border="0", command=browse_function)
        upload_button.pack(side=LEFT, padx=20, pady=15)

        space_label3 = Label(mid_frame1, bg="gray")
        space_label3.pack(side=LEFT, fill="x", padx=30)

        # eticheta de cale + spațiul de intrare a etichetei
        path_label = Label(mid_frame1, text="Path:", font=("Arial", 15, "bold"), width=5,
                           bg="gray", foreground="White")
        path_label.pack(side=LEFT, fill='x')

        path_entry = Entry(mid_frame1, textvariable=path_text, font=("Arial", 15, "bold"), width=45,
                           bg="white", foreground="black", state="disabled")
        path_entry.pack(side=LEFT, padx=5)

        # se adaugă un buton de predicție
        predict_button = tk.Button(self, text="Get Prediction", font=("Arial", 20), bg="white", width=10,
                                   height=2, border="0", command=get_prediction)
        predict_button.pack(pady=10)

        mid_frame2 = tk.Frame(self,  relief='raised', bg="gray")
        mid_frame2.pack()

        canvas = tk.Label(mid_frame2, width=33, height=15, bg="gray33")
        canvas.grid(row=0, column=0, rowspan=3)

        space_label4 = Label(mid_frame2, bg="gray", width=10)
        space_label4.grid(row=0, column=1, rowspan=3)

        # se adaugă etichetele și zonele de text
        # pentru TI-RADS și factorul de încredere
        ti_rads_label = Label(mid_frame2, text="TI-RADS:", font=("Arial", 15, "bold"),
                              bg="gray", foreground="White")
        ti_rads_label.grid(row=0, column=2)

        ti_rads_entry = Entry(mid_frame2, font=("Arial", 15, "bold"), textvariable=ti_rads_text, width=20,
                              bg="white", foreground="black", state="disabled")
        ti_rads_entry.grid(row=0, column=3)

        trust_ti_rads_label = Label(mid_frame2, text="Trust TI-RADS:", font=("Arial", 15, "bold"),
                                    bg="gray", foreground="White")
        trust_ti_rads_label.grid(row=1, column=2)

        trust_ti_rads_entry = Entry(mid_frame2, font=("Arial", 15, "bold"), textvariable=trust_ti_rads, width=20,
                                    bg="white", foreground="black", state="disabled")
        trust_ti_rads_entry.grid(row=1, column=3)

        # tipul nodulului și factorul de încredere
        type_label = Label(mid_frame2, text="TYPE:", font=("Arial", 15, "bold"),
                           bg="gray", foreground="White")
        type_label.grid(row=2, column=2)

        type_entry = Entry(mid_frame2, font=("Arial", 15, "bold"), textvariable=nodule_type, width=20,
                           bg="white", foreground="black", state="disabled")
        type_entry.grid(row=2, column=3)

        trust_label = Label(mid_frame2, text="Trust TYPE:", font=("Arial", 15, "bold"),
                            bg="gray", foreground="White")
        trust_label.grid(row=3, column=2)

        trust_entry = Entry(mid_frame2, font=("Arial", 15, "bold"), textvariable=trust, width=20,
                            bg="white", foreground="black", state="disabled")
        trust_entry.grid(row=3, column=3)

        # se adaugă frame-ul din partea de jos a aplicației
        bottom_frame = tk.Frame(self, relief='raised', borderwidth=0, bg="gray33")
        bottom_frame.pack(fill='x', side=BOTTOM)

        space_label5 = tk.Label(bottom_frame, text='', font=("Arial", 30), bg="gray33")
        space_label5.pack()


# Se creează a partea de segmentare
class PageTwo(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, bg="gray")
        self.controller = controller

        # se inițializează variabilele
        trust = StringVar()
        path_text = StringVar()

        # dimensiunea imaginii pentru modelul de segmentare
        img_width = 256
        img_height = 256
        img_channels = 3

        # dimensiunea imaginii pentru modelul de diferențiere
        height = 224
        width = 224

        # funcția de căutare
        def browse_function():
            filepath = filedialog.askopenfilename()
            path_text.set(filepath)
            image_display()

        # funcția de afișare
        def image_display():
            img = Image.open('%s' % str(path_text.get()))
            resize_img = img.resize((300, 250))
            image = ImageTk.PhotoImage(resize_img)
            canvas.configure(image=image, width=0, height=0)
            canvas.image = image

        # funcția de predicție
        def get_prediction():
            x_test = np.zeros((1, height, width, img_channels), dtype=np.uint8)
            test = np.zeros((1, img_height, img_width, img_channels), dtype=np.uint8)

            # se încarcă modelele
            us = tf.keras.models.load_model('model_us_alexnet.h5')
            model = tf.keras.models.load_model('model_FCN_ResNet50_bun3.h5')
            with open('fcn_resnet50.pkl', 'rb') as f:
                fcn_resnet_history = pickle.load(f)

            acc_fcn_resnet = fcn_resnet_history['accuracy']
            acc_fcn_resnet = np.mean(acc_fcn_resnet)

            img = imread(str(path_text.get()))[:, :, :img_channels]
            img = resize(img, (width, height, img_channels), mode='constant', preserve_range=True)
            x_test[0] = img

            img = imread(str(path_text.get()))[:, :, :img_channels]
            img = resize(img, (img_width, img_height, img_channels), mode='constant', preserve_range=True)
            test[0] = img

            # se generează predicțiile
            predictions_us = us.predict(x_test, verbose=1)
            predicted_classes_us = np.argmax(predictions_us, axis=1)

            predictions = model.predict(test, verbose=1)
            pred_test_t = (predictions > 0.5).astype(np.uint8)

            # se afișează imaginea segmentării generate
            f = Figure(figsize=(2.6, 2.6), dpi=100)
            a = f.add_subplot(111)

            a.imshow(np.squeeze(pred_test_t[0]), extent=[0, 100, 0, 1], aspect='auto')
            a.axis('off')
            a.set_position([0, 0, 1, 1])
            canvas2 = FigureCanvasTkAgg(f, mid_frame2)

            # dacă este o imagine ecografică se va afișa imaginea
            if predicted_classes_us == 0:
                canvas2.draw()
                canvas2.get_tk_widget().grid(row=0, column=3, rowspan=4)

                space_label6.configure(height=0, width=0)

                acc_fcn_resnet = acc_fcn_resnet * 100
                trust.set(str("%.2f" % acc_fcn_resnet)+"%")

            # dacă nu este o imagine ecografică se va afișa o figură goală
            else:
                tkinter.messagebox.showinfo('Error Message', 'Error: Please select a US image!')
                f.clear()
                a = f.add_subplot(111)
                a.clear()
                a.set_position([0, 0, 1, 1])
                canvas2 = FigureCanvasTkAgg(f, mid_frame2)
                canvas2.draw()
                canvas2.get_tk_widget().grid(row=0, column=3, rowspan=4)
                trust.set("NULL")

        # se adaugă spații
        space_label2 = Label(self, bg="gray")
        space_label2.pack(side=TOP, fill="x", pady=2)

        back_frame = tk.Frame(self, relief='raised', borderwidth=0, bg="gray")
        back_frame.pack(fill='x', side=TOP)

        # se adaugă butonul de mers înapoi
        go_back1 = tk.Button(back_frame, text="Go Back", font=("Arial", 15), bg="white", width=10, height=2, border="0",
                             command=lambda: controller.show_frame("StartPage"))
        go_back1.pack(side=LEFT, padx=10)

        # titlul paginii de segmentare
        segmentation_label = Label(self, text="Segmentation", font=("Arial", 30, "bold"), width=10,
                                   bg="gray", foreground="White", justify=LEFT)
        segmentation_label.pack(side=TOP, fill="x", pady=5)

        # se împarte în mai multe părți/frame-uri
        mid_frame1 = tk.Frame(self, relief='raised', borderwidth=0, bg="gray")
        mid_frame1.pack(fill='x')

        # se adaugă butonul de încărcare a imaginii
        upload_button = tk.Button(mid_frame1, text="Upload the image", font=("Arial", 15), bg="white", width=15,
                                  height=2, border="0", command=browse_function)
        upload_button.pack(side=LEFT, padx=20, pady=15)

        # eticheta + spațiul de text pentru cale
        path_label = Label(mid_frame1, text="Path:", font=("Arial", 15, "bold"), width=15,
                           bg="gray", foreground="White")
        path_label.pack(side=LEFT, fill='x', pady=10)

        path_entry = Entry(mid_frame1, textvariable=path_text, font=("Arial", 15, "bold"), width=45,
                           bg="white", foreground="black", state="disabled")
        path_entry.pack(side=LEFT, padx=5)

        # butonul care generează predicțiile
        predict_button = tk.Button(self, text="Get Prediction", font=("Arial", 20), bg="white", width=10,
                                   height=2, border="0", command=get_prediction)
        predict_button.pack(pady=10)

        # se trece în al doilea frame
        mid_frame2 = tk.Frame(self, relief='raised', borderwidth=0, bg="gray")
        mid_frame2.pack(fill='x')

        space_label4 = tk.Label(mid_frame2, width=1, bg="gray")
        space_label4.grid(row=0, column=0, rowspan=4)

        # se afișează imaginea selectată
        canvas = tk.Label(mid_frame2, borderwidth=3, width=33, height=15, bg="gray33")
        canvas.grid(row=0, column=1, rowspan=4)

        space_label5 = tk.Label(mid_frame2, width=1, bg="gray")
        space_label5.grid(row=0, column=4, rowspan=3)

        # se afișează săgeata care desparte imaginea selectată de predicție
        image_photo = tk.PhotoImage(file='right-arrow.png')

        # se afișează imaginea generată
        image_label = tk.Label(mid_frame2, image=image_photo, bg="gray")
        image_label.image = image_photo
        image_label.grid(row=0, column=2, rowspan=4)

        space_label6 = tk.Label(mid_frame2, width=33, height=15, bg="gray33")
        space_label6.grid(row=0, column=3, rowspan=4)

        # eticheta + locul pentru text al încrederii modelului
        trust_label = Label(mid_frame2, text="Trust:", font=("Arial", 15, "bold"),
                            bg="gray", foreground="White")
        trust_label.grid(row=0, column=5)

        trust_entry = Entry(mid_frame2, font=("Arial", 15, "bold"), textvariable=trust, width=10,
                            bg="white", foreground="black", state="disabled")
        trust_entry.grid(row=1, column=5)

        space_label7 = tk.Label(mid_frame2, width=10, bg="gray")
        space_label7.grid(row=4, column=5)

        # se adaugă frame-ul gri din partea de jos a aplicației
        bottom_frame = tk.Frame(self, relief='raised', borderwidth=0, bg="gray33")
        bottom_frame.pack(fill='x', side=BOTTOM)

        color_label6 = tk.Label(bottom_frame, text='', font=("Arial", 30), bg="gray33")
        color_label6.pack()


#
if __name__ == "__main__":
    app = App()
    app.mainloop()
