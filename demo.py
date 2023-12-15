import sys, random, os
import imageio.v2 as imageio
import numpy as np
from cryp import RMT,AES
from Neuracrypt import NeuraCrypt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QStackedWidget,QDialog,
                             QHBoxLayout, QGridLayout, QLabel, QRadioButton, QComboBox, QFileDialog, QSlider, QTextEdit, QMessageBox, QFileDialog, QInputDialog)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from torchvision import datasets
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from PIL import Image


def save_image_from_array(img_array, save_path):
    """
    Save a numpy array as an image using PIL.

    Args:
        img_array (numpy.ndarray): The image data in numpy array format.
        save_path (str): The path to save the image.
    """
    img = Image.fromarray(img_array.astype('uint8'))
    img.save(save_path, 'JPEG')



def load_image_to_array(image_path):
    """
    Load an image from a file path and return it as a numpy array.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The image data in numpy array format.
    """
    with Image.open(image_path) as img:
        img_array = np.array(img, dtype=np.float32)
    return img_array

class ImageDisguisingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Main layout container
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Setup canvas switcher buttons
        self.setup_canvas_switcher()

        # Set up the initial canvas (canvas_one is shown by default)
        self.canvas_one = QWidget()
        self.canvas_two = QWidget()
        self.main_layout.addWidget(self.canvas_one)
        self.main_layout.addWidget(self.canvas_two)
        self.canvas_two.hide()  # Initially hide canvas_two

        # Setup canvases with their respective UI elements
        self.setup_canvas_one()
          # Call setup_canvas_two if you have anything to add there
        self.setup_canvas_two()

        # Set the window title and size
        self.setWindowTitle("Image Disguising: Client-Side Demo System")
        self.setGeometry(100, 100, 800, 600)

    def setup_canvas_switcher(self):
        # Header section with canvas switcher buttons
        header_layout = QHBoxLayout()
        btn_canvas_one = QPushButton("DisguisedNet")
        btn_canvas_two = QPushButton("NeuraCrypt")
        btn_canvas_one.clicked.connect(self.show_canvas_one)
        btn_canvas_two.clicked.connect(self.show_canvas_two)
        header_layout.addWidget(btn_canvas_one)
        header_layout.addWidget(btn_canvas_two)
        self.main_layout.addLayout(header_layout)

    def setup_canvas_one(self):
        layout = QVBoxLayout(self.canvas_one)

        # Add your UI components to the canvas_one layout
        # Encryption and Attack Settings
        settings_layout = QHBoxLayout()

        # Encryption settings section
        encryption_layout = QVBoxLayout()
        lbl_method = QLabel("Method")
        self.rb_rmt = QRadioButton("RMT")
        self.rb_aes = QRadioButton("AES")
        lbl_block_size = QLabel("Block Size")
        self.rb_aes.toggled.connect(self.on_aes_toggled)
        self.block_size_dropdown = QComboBox()
        self.block_size_dropdown.addItems(['2', '4', '8', '16', '32', '64'])

        encryption_layout.addWidget(lbl_method)
        encryption_layout.addWidget(self.rb_rmt)
        encryption_layout.addWidget(self.rb_aes)
        encryption_layout.addWidget(lbl_block_size)
        encryption_layout.addWidget(self.block_size_dropdown)

        self.noise_slider_lbl = QLabel("Noise Level: 0")
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setMinimum(0)
        self.noise_slider.setMaximum(100)
        self.noise_slider.setValue(0)
        self.noise_slider.setTickInterval(1)
        self.noise_slider.setTickPosition(QSlider.TicksBelow)
        self.noise_slider.valueChanged.connect(self.update_noise_label)
        encryption_layout.addWidget(self.noise_slider_lbl)
        encryption_layout.addWidget(self.noise_slider)

        # Placeholder for Attack settings section
        attack_layout = QVBoxLayout()
        lbl_attack_settings = QLabel("Attack Settings")  # Placeholder label
        attack_layout.addWidget(lbl_attack_settings)

        self.knownpairs_label = QLabel("Number of known pairs: 0")
        self.knownpairs_slider = QSlider(Qt.Horizontal)
        self.knownpairs_slider.setMinimum(0)
        self.knownpairs_slider.setMaximum(20)  # Assuming you have self.original_images initialized
        self.knownpairs_slider.valueChanged.connect(self.update_knownpairs_label)
        attack_layout.addWidget(self.knownpairs_label)
        attack_layout.addWidget(self.knownpairs_slider)

        results_show_layout = QVBoxLayout()

        # Create a button and add it to the layout
        button_show_image = QPushButton("Show results")
        results_show_layout.addWidget(button_show_image)

        # Connect the button's clicked signal to the slot method
        button_show_image.clicked.connect(self.on_show_image_clicked)
        settings_layout.addLayout(encryption_layout)
        settings_layout.addStretch(1)
        settings_layout.addLayout(attack_layout)
        layout.addLayout(settings_layout)
        layout.addLayout(results_show_layout)


        # Sample Images section
        sample_images_layout = QHBoxLayout()
        self.lbl_original = QLabel()
        self.lbl_original.setFixedSize(200, 200)
        self.lbl_original.setStyleSheet("border: 1px solid black")
        self.lbl_disguised = QLabel("Disguised")
        self.lbl_disguised.setFixedSize(200, 200)
        self.lbl_disguised.setStyleSheet("border: 1px solid black")
        self.lbl_attack_reconstructed = QLabel("Attack Reconstructed")
        self.lbl_attack_reconstructed.setFixedSize(200, 200)
        self.lbl_attack_reconstructed.setStyleSheet("border: 1px solid black")
        sample_images_layout.addWidget(self.lbl_original)
        sample_images_layout.addWidget(self.lbl_disguised)
        sample_images_layout.addWidget(self.lbl_attack_reconstructed)
        layout.addLayout(sample_images_layout)

        # Control Panel section
        control_panel_layout = QVBoxLayout()
        button_upload = QPushButton('Upload Image', self)
        button_upload.clicked.connect(self.openFileNameDialog)
        button_encrypt = QPushButton('Encrypt!', self)
        button_encrypt.clicked.connect(self.on_enrypt_click)
        self.button_attack = QPushButton('Attack!', self)
        self.button_attack.clicked.connect(self.on_attack_click)
        control_panel_layout.addWidget(button_upload)
        control_panel_layout.addWidget(button_encrypt)
        control_panel_layout.addWidget(self.button_attack)
        layout.addLayout(control_panel_layout)
        self.chatbox = QTextEdit(self)
        self.chatbox.setReadOnly(True)
        layout.addWidget(self.chatbox)
        self.setWindowTitle("Image Disguising: Client-Side Demo System")
        self.setGeometry(100, 100, 800, 600)
        # At the end of this method,set the layout to the canvas_one widget
        self.canvas_one.setLayout(layout)

    def setup_canvas_two(self):
        # Add UI components to the canvas_one layout
        layout = QVBoxLayout(self.canvas_two)

        # Encryption and Attack Settings
        settings_layout = QHBoxLayout()

        # Encryption settings section
        encryption_layout = QVBoxLayout()
        lbl_block_size = QLabel("Block Size")
        self.block_size_dropdown_2 = QComboBox()
        self.block_size_dropdown_2.addItems(['2', '4', '8', '16', '32', '64'])
        encryption_layout.addWidget(lbl_block_size)
        encryption_layout.addWidget(self.block_size_dropdown_2)
        settings_layout.addLayout(encryption_layout)
        settings_layout.addStretch(1)
        layout.addLayout(settings_layout)
        results_show_layout = QVBoxLayout()

        # Create a button and add it to the layout
        button_show_image = QPushButton("Show results")
        results_show_layout.addWidget(button_show_image)

        # Connect the button's clicked signal to the slot method
        button_show_image.clicked.connect(self.on_show_image_clicked_neura)
        layout.addLayout(results_show_layout)

        # Sample Images section
        sample_images_layout = QHBoxLayout()
        self.lbl_original_2 = QLabel()
        self.lbl_original_2.setFixedSize(200, 200)
        self.lbl_original_2.setStyleSheet("border: 1px solid black")
        self.lbl_disguised_2 = QLabel("Disguised")
        self.lbl_disguised_2.setFixedSize(200, 200)
        self.lbl_disguised_2.setStyleSheet("border: 1px solid black")
        sample_images_layout.addWidget(self.lbl_original_2)
        sample_images_layout.addWidget(self.lbl_disguised_2)
        layout.addLayout(sample_images_layout)

        # Control Panel section
        control_panel_layout = QVBoxLayout()
        button_upload = QPushButton('Upload Image', self)
        button_upload.clicked.connect(self.openFileNameDialog)
        button_encrypt = QPushButton('Encrypt!', self)
        button_encrypt.clicked.connect(self.on_neuracrypt_click)
        control_panel_layout.addWidget(button_upload)
        control_panel_layout.addWidget(button_encrypt)
        layout.addLayout(control_panel_layout)
        self.chatbox_2 = QTextEdit(self)
        self.chatbox_2.setReadOnly(True)
        layout.addWidget(self.chatbox_2)
        self.setWindowTitle("NeuraCrypt: Client-Side Demo System")
        self.setGeometry(100, 100, 800, 600)

        # At the end of this method, set the layout to the canvas_one widget
        self.canvas_two.setLayout(layout)

    def on_show_image_clicked(self):
        # Slot method to handle button click
        # Create the dialog
        image_dialog = QDialog(self)
        image_dialog.setWindowTitle("Image Viewer")

        # Create a label and set the pixmap with the image you want to show
        label = QLabel(image_dialog)
        pixmap = QPixmap("./results/r1.png")
        label.setPixmap(pixmap)

        # Adjust the dialog size to fit image or make other adjustments
        image_dialog.resize(pixmap.width(), pixmap.height())

        # Layout for the dialog
        dialog_layout = QVBoxLayout()
        dialog_layout.addWidget(label)
        image_dialog.setLayout(dialog_layout)

        # Show the dialog
        image_dialog.exec_()

    def on_show_image_clicked_neura(self):
        # Slot method to handle button click
        # Create the dialog
        image_dialog = QDialog(self)
        image_dialog.setWindowTitle("Image Viewer")

        # Create a label and set the pixmap with the image you want to show
        label = QLabel(image_dialog)
        pixmap = QPixmap("./results/r2.png")
        label.setPixmap(pixmap)

        # Adjust the dialog size to fit image or make other adjustments
        image_dialog.resize(pixmap.width(), pixmap.height())

        # Layout for the dialog
        dialog_layout = QVBoxLayout()
        dialog_layout.addWidget(label)
        image_dialog.setLayout(dialog_layout)

        # Show the dialog
        image_dialog.exec_()

    def on_aes_toggled(self):
        if self.rb_aes.isChecked():
            # Disable the button when rb_rmt is selected (checked)
            self.button_attack.setEnabled(False)
            self.block_size_dropdown.clear()
            self.block_size_dropdown.addItems(['4', '8', '16', '32', '64'])
        else:
            # Enable the button back when rb_rmt is not selected
            self.button_attack.setEnabled(True)

    def show_canvas_one(self):
        self.canvas_two.hide()
        self.canvas_one.show()

    def show_canvas_two(self):
        self.canvas_one.hide()
        self.canvas_two.show()

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.chatbox.append("Please choose a image to encrypt. Click cancel to choose a directory of images")

        # Try getting a file
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                  "All Files (*);;Images (*.png *.jpg *.bmp)", options=options)

        if filePath:
            self.button_attack.setEnabled(False)

        # If no file is selected, try getting a directory
        if not filePath:
            self.button_attack.setEnabled(True)
            self.chatbox.append("Please choose a directory of images.")
            filePath = QFileDialog.getExistingDirectory(self, "Open Directory", "", options=options)

            # Check if the selected path is a directory
            if os.path.isdir(filePath):

                # Get a list of all the image files in the directory
                self.image_files = [f for f in os.listdir(filePath) if
                                    os.path.isfile(os.path.join(filePath, f)) and f.lower().endswith(
                                        ('.png', '.jpg', '.bmp'))]

                if not self.image_files:
                    self.chatbox.append("The directory contains no valid image files.")
                    return
                    # Ask the user which image they want to show
                image_to_show, ok = QInputDialog.getItem(self, "Select an image", "Choose an image to display:",
                                                         self.image_files, 0, False)
                self.image_to_show = image_to_show
                index = 0
                for i in self.image_files:
                    if self.image_to_show == i:
                        self.index = index
                    else:
                        index += 1
                if ok:
                    pixmap = QPixmap(os.path.join(filePath, image_to_show))
                    self.lbl_original.setPixmap(pixmap.scaled(200, 200))
                    self.lbl_original_2.setPixmap(pixmap.scaled(200, 200))

                    # Store path for future encryption
                    self.images_directory = filePath

                    # If it's just a file
        else:
            pixmap = QPixmap(filePath)
            self.lbl_original.setPixmap(pixmap.scaled(200, 200))
            self.lbl_original_2.setPixmap(pixmap.scaled(200, 200))
            self.original_image = imageio.imread(filePath)
            self.image_files = None

    def on_enrypt_click(self):

        if self.image_files != None:

            image_files = self.image_files

            self.encrypted_images = []

            self.original_images = []
            # Ask the user for a directory to save encrypted images
            self.chatbox.append("Encrypting: noise = " + str(self.noise_slider.value()) + ", Block size = " + str(
                self.block_size_dropdown.currentText()))

            self.chatbox.append("Please choose a place to store the encrypted images")

            options = QFileDialog.Options()

            encrypted_images_save_path = QFileDialog.getExistingDirectory(self,
                                                                          "Choose Directory to Save Encrypted Images",
                                                                          "", options=options)

            if not encrypted_images_save_path:
                # User canceled the save location dialog
                return

            # Pad original image if neccessary
            image_temp = load_image_to_array(os.path.join(self.images_directory, image_files[0]))

            # If adjusted row or col different from the original image, need padding
            adjusted_row = (image_temp.shape[0] + int(self.block_size_dropdown.currentText()) - 1) // int(
                self.block_size_dropdown.currentText()) * int(self.block_size_dropdown.currentText())

            adjusted_col = (image_temp.shape[1] + int(self.block_size_dropdown.currentText()) - 1) // int(
                self.block_size_dropdown.currentText()) * int(self.block_size_dropdown.currentText())

            # Generate padding
            if image_temp.shape[0] != adjusted_row or image_temp.shape[1] != adjusted_col:
                pad_row = adjusted_row - image_temp.shape[0]

                pad_col = adjusted_col - image_temp.shape[1]

            else:
                pad_row = 0

                pad_col = 0

            if len(image_temp.shape)==3:

                if self.rb_rmt.isChecked():

                    self.encoder = RMT(image_size=(image_temp.shape[0] + pad_row, image_temp.shape[1] + pad_col,image_temp.shape[2]),
                                   block_size=int(self.block_size_dropdown.currentText()),
                                   Shuffle=False)

                elif self.rb_aes.isChecked():

                    self.encoder = AES(image_size=(image_temp.shape[0] + pad_row, image_temp.shape[1] + pad_col,image_temp.shape[2]),
                                           block_size=(int(self.block_size_dropdown.currentText()),int(self.block_size_dropdown.currentText())))

                else:
                    self.chatbox.append('Please choose from AES and RMT.')

            else:
                if self.rb_rmt.isChecked():

                    self.encoder = RMT(
                        image_size=(image_temp.shape[0] + pad_row, image_temp.shape[1] + pad_col),
                        block_size=int(self.block_size_dropdown.currentText()),
                        Shuffle=False)

                elif self.rb_aes.isChecked():

                    self.encoder = AES(
                        image_size=(image_temp.shape[0] + pad_row, image_temp.shape[1] + pad_col),
                        block_size=(
                        int(self.block_size_dropdown.currentText()), int(self.block_size_dropdown.currentText())))

                else:
                    self.chatbox.append('Please choose from AES and RMT.')


            for image_file in image_files:

                img_path = os.path.join(self.images_directory, image_file)

                image = load_image_to_array(img_path)

                # Apply padding

                if image_temp.shape[0] != adjusted_row or image_temp.shape[1] != adjusted_col:

                    if len(image.shape) == 3:  # RGB image

                        image_padded = np.pad(image, ((0, pad_row), (0, pad_col), (0, 0)), mode='edge')

                    else:  # Grayscale image

                        image_padded = np.pad(image, ((0, pad_row), (0, pad_col)), mode='edge')

                else:

                    image_padded = image

                self.original_images.append(image_padded)

                noise_level = self.noise_slider.value()

                if noise_level == 0:

                    noise = False

                    noise_level = 1

                else:

                    noise = True

                encrypted_img_array = self.encoder.Encode(image_padded, noise=noise, noise_level=noise_level)

                self.encrypted_images.append(encrypted_img_array)

                encrypted_image_path = os.path.join(encrypted_images_save_path, "encrypted_" + image_file)

                save_image_from_array(encrypted_img_array, encrypted_image_path)

                # Check if it's the image selected to display
                if image_file == self.image_to_show:

                    if len(encrypted_img_array.shape) == 3:  # RGB Image

                        height, width, channel = encrypted_img_array.shape

                        bytesPerLine = 3 * width

                        qImg = QImage(np.clip(encrypted_img_array, 0, 255).astype(np.uint8).data, width, height,
                                      bytesPerLine, QImage.Format_RGB888)

                    else:  # Greyscale Image

                        height, width = encrypted_img_array.shape

                        bytesPerLine = width

                        qImg = QImage(np.clip(encrypted_img_array, 0, 255).astype(np.uint8).data, width, height,
                                      bytesPerLine, QImage.Format_Grayscale8)

                    encrypted_pixmap = QPixmap.fromImage(qImg)

                    self.lbl_disguised.setPixmap(encrypted_pixmap.scaled(200, 200))
                    self.lbl_disguised_2.setPixmap(encrypted_pixmap.scaled(200, 200))

            self.chatbox.append("Encryption done for all images in the directory!")
            self.chatbox_2.append("Encryption done for all images in the directory!")

        else:

            self.chatbox.append("Encrypting: noise = " + str(self.noise_slider.value()) + ", Block size = " + str(
                self.block_size_dropdown.currentText()))
            self.chatbox_2.append("Encrypting: noise = " + str(self.noise_slider.value()) + ", Block size = " + str(
                self.block_size_dropdown.currentText()))

            # Fetch the noise level from the slider
            # The slider widget has not been defined in the initial provided code,
            noise_level = self.noise_slider.value()

            if noise_level == 0:

                noise = False

                noise_level = 1
            else:
                noise = True

            self.chatbox.append("Please store the encrypted image")

            options = QFileDialog.Options()

            filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                      "PNG Files (*.png);;JPG Files (*.jpg);;All Files (*)",
                                                      options=options)

            # Initializing the RMT encoder with the specified parameters

            image_temp = self.original_image

            # If adjusted row or col different from the original image, need padding
            adjusted_row = (image_temp.shape[0] + int(self.block_size_dropdown.currentText()) - 1) // int(
                self.block_size_dropdown.currentText()) * int(self.block_size_dropdown.currentText())

            adjusted_col = (image_temp.shape[1] + int(self.block_size_dropdown.currentText()) - 1) // int(
                self.block_size_dropdown.currentText()) * int(self.block_size_dropdown.currentText())

            # Generate padding
            if image_temp.shape[0] != adjusted_row or image_temp.shape[1] != adjusted_col:
                pad_row = adjusted_row - image_temp.shape[0]

                pad_col = adjusted_col - image_temp.shape[1]

            else:
                pad_row = 0
                pad_col = 0

            if len(image_temp.shape)==3:

                if self.rb_rmt.isChecked():

                    self.encoder = RMT(
                        image_size=(image_temp.shape[0] + pad_row, image_temp.shape[1] + pad_col, image_temp.shape[2]),
                        block_size=int(self.block_size_dropdown.currentText()),
                        Shuffle=False)

                elif self.rb_aes.isChecked():

                    self.encoder = AES(
                        image_size=(image_temp.shape[0] + pad_row, image_temp.shape[1] + pad_col, image_temp.shape[2]),
                        block_size=(
                        int(self.block_size_dropdown.currentText()), int(self.block_size_dropdown.currentText())))

                else:
                    self.chatbox.append('Please choose from AES and RMT.')
                    self.chatbox_2.append('Please choose from AES and RMT.')
            else:

                if self.rb_rmt.isChecked():

                    self.encoder = RMT(
                        image_size=(image_temp.shape[0] + pad_row, image_temp.shape[1] + pad_col),
                        block_size=int(self.block_size_dropdown.currentText()),
                        Shuffle=False)

                elif self.rb_aes.isChecked():

                    self.encoder = AES(
                        image_size=(image_temp.shape[0] + pad_row, image_temp.shape[1] + pad_col),
                        block_size=(
                        int(self.block_size_dropdown.currentText()), int(self.block_size_dropdown.currentText())))

                else:
                    self.chatbox.append('Please choose from AES and RMT.')

            if image_temp.shape[0] != adjusted_row or image_temp.shape[1] != adjusted_col:

                if len(image_temp.shape) == 3:  # RGB image

                    image_padded = np.pad(image_temp, ((0, pad_row), (0, pad_col), (0, 0)), mode='edge')

                else:  # Grayscale image

                    image_padded = np.pad(image_temp, ((0, pad_row), (0, pad_col)), mode='edge')

            else:

                image_padded = image_temp


            # Encrypting the original image using the RMT encoder
            encrypted_img_array = self.encoder.Encode(image_padded, noise=noise, noise_level=noise_level)

            if filePath:
                save_image_from_array(encrypted_img_array, filePath)
            # Convert the encoded image back to QPixmap and display on the label

            if len(encrypted_img_array.shape) == 3:  # RGB Image

                height, width, channel = encrypted_img_array.shape

                bytesPerLine = 3 * width

                qImg = QImage(np.clip(encrypted_img_array, 0, 255).astype(np.uint8).data, width, height, bytesPerLine,
                              QImage.Format_RGB888)

            else:  # Greyscale Image

                height, width = encrypted_img_array.shape

                bytesPerLine = width

                qImg = QImage(np.clip(encrypted_img_array, 0, 255).astype(np.uint8).data, width, height, bytesPerLine,
                              QImage.Format_Grayscale8)

            encrypted_pixmap = QPixmap.fromImage(qImg)

            self.lbl_disguised.setPixmap(encrypted_pixmap.scaled(200, 200))
            self.lbl_disguised_2.setPixmap(encrypted_pixmap.scaled(200, 200))

            self.chatbox.append("Encryption done!")
            self.chatbox_2.append("Encryption done!")

    def on_neuracrypt_click(self):

        if self.image_files != None:

            image_files = self.image_files

            self.encrypted_images = []

            self.original_images = []
            # Ask the user for a directory to save encrypted images
            self.chatbox.append("Encrypting: noise = " + str(self.noise_slider.value()) + ", Block size = " + str(
                self.block_size_dropdown_2.currentText()))
            self.chatbox_2.append("Encrypting: noise = " + str(self.noise_slider.value()) + ", Block size = " + str(
                self.block_size_dropdown_2.currentText()))

            self.chatbox.append("Please choose a place to store the encrypted images")
            self.chatbox_2.append("Please choose a place to store the encrypted images")

            options = QFileDialog.Options()

            encrypted_images_save_path = QFileDialog.getExistingDirectory(self,
                                                                          "Choose Directory to Save Encrypted Images",
                                                                          "", options=options)

            if not encrypted_images_save_path:
                # User canceled the save location dialog
                return

            # Pad original image if neccessary
            image_temp = load_image_to_array(os.path.join(self.images_directory, image_files[0]))

            # If adjusted row or col different from the original image, need padding
            adjusted_row = (image_temp.shape[0] + int(self.block_size_dropdown_2.currentText()) - 1) // int(
                self.block_size_dropdown_2.currentText()) * int(self.block_size_dropdown_2.currentText())

            adjusted_col = (image_temp.shape[1] + int(self.block_size_dropdown_2.currentText()) - 1) // int(
                self.block_size_dropdown_2.currentText()) * int(self.block_size_dropdown_2.currentText())

            # Generate padding
            if image_temp.shape[0] != adjusted_row or image_temp.shape[1] != adjusted_col:
                pad_row = adjusted_row - image_temp.shape[0]

                pad_col = adjusted_col - image_temp.shape[1]

            else:
                pad_row = 0

                pad_col = 0

            noise_level = self.noise_slider.value()

            if noise_level == 0:

                noise = False

                noise_level = 1

            else:

                noise = True

            if len(image_temp.shape)==3:

                self.neuracrypt = NeuraCrypt(image_size=(image_temp.shape[0] + pad_row, image_temp.shape[1] + pad_col,image_temp.shape[2]),
                                   patch_size=int(self.block_size_dropdown_2.currentText()),
                                   noise = noise,
                                   noise_level = self.noise_slider.value())

            else:
                self.neuracrypt = NeuraCrypt(
                    image_size=(image_temp.shape[0] + pad_row, image_temp.shape[1] + pad_col),
                    patch_size=int(self.block_size_dropdown_2.currentText()),
                    noise=noise,
                    noise_level=self.noise_slider.value())


            for image_file in image_files:

                img_path = os.path.join(self.images_directory, image_file)

                image = load_image_to_array(img_path)

                # Apply padding

                if image_temp.shape[0] != adjusted_row or image_temp.shape[1] != adjusted_col:

                    if len(image.shape) == 3:  # RGB image

                        image_padded = np.pad(image, ((0, pad_row), (0, pad_col), (0, 0)), mode='edge')

                    else:  # Grayscale image

                        image_padded = np.pad(image, ((0, pad_row), (0, pad_col)), mode='edge')

                else:

                    image_padded = image

                self.original_images.append(image_padded)

                encrypted_img_array = self.neuracrypt.forward(image_padded).detach().numpy()

                self.encrypted_images.append(encrypted_img_array)

                encrypted_image_path = os.path.join(encrypted_images_save_path, "encrypted_" + image_file)

                save_image_from_array(encrypted_img_array, encrypted_image_path)

                # Check if it's the image selected to display
                if image_file == self.image_to_show:

                    if len(encrypted_img_array.shape) == 3:  # RGB Image

                        height, width, channel = encrypted_img_array.shape

                        encrypted_img_bytes = bytes(encrypted_img_array)

                        bytesPerLine = 3 * width

                        qImg = QImage(encrypted_img_bytes, width, height,
                                      bytesPerLine, QImage.Format_RGB888)

                    else:  # Greyscale Image

                        height, width = encrypted_img_array.shape

                        bytesPerLine = width

                        qImg = QImage(encrypted_img_array.data, width, height,
                                      bytesPerLine, QImage.Format_Grayscale8)

                    encrypted_pixmap = QPixmap.fromImage(qImg)

                    self.lbl_disguised.setPixmap(encrypted_pixmap.scaled(200, 200))
                    self.lbl_disguised_2.setPixmap(encrypted_pixmap.scaled(200, 200))

            self.chatbox.append("Encryption done for all images in the directory!")
            self.chatbox_2.append("Encryption done for all images in the directory!")

        else:

            self.chatbox.append("Encrypting: noise = " + str(self.noise_slider.value()) + ", Block size = " + str(
                self.block_size_dropdown_2.currentText()))
            self.chatbox_2.append("Encrypting: noise = " + str(self.noise_slider.value()) + ", Block size = " + str(
                self.block_size_dropdown_2.currentText()))

            # Fetch the noise level from the slider
            # The slider widget has not been defined in the initial provided code,
            # so make sure you have added a QSlider and named it 'noise_slider' in the GUI.
            noise_level = self.noise_slider.value()

            if noise_level == 0:

                noise = False

                noise_level = 1
            else:
                noise = True

            self.chatbox_2.append("Please store the encrypted image")

            options = QFileDialog.Options()

            filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                      "PNG Files (*.png);;JPG Files (*.jpg);;All Files (*)",
                                                      options=options)

            # Initializing the RMT encoder with the specified parameters
            # Make sure you've added a dropdown menu for block size and named it 'block_size_dropdown'
            image_temp = self.original_image

            # If adjusted row or col different from the original image, need padding
            adjusted_row = (image_temp.shape[0] + int(self.block_size_dropdown_2.currentText()) - 1) // int(
                self.block_size_dropdown_2.currentText()) * int(self.block_size_dropdown_2.currentText())

            adjusted_col = (image_temp.shape[1] + int(self.block_size_dropdown_2.currentText()) - 1) // int(
                self.block_size_dropdown_2.currentText()) * int(self.block_size_dropdown_2.currentText())

            # Generate padding
            if image_temp.shape[0] != adjusted_row or image_temp.shape[1] != adjusted_col:
                pad_row = adjusted_row - image_temp.shape[0]

                pad_col = adjusted_col - image_temp.shape[1]

            else:
                pad_row = 0

                pad_col = 0

            if image_temp.shape[0] != adjusted_row or image_temp.shape[1] != adjusted_col:

                if len(image_temp.shape) == 3:  # RGB image

                    image_padded = np.pad(image_temp, ((0, pad_row), (0, pad_col), (0, 0)), mode='edge')

                else:  # Grayscale image

                    image_padded = np.pad(image_temp, ((0, pad_row), (0, pad_col)), mode='edge')

            else:

                image_padded = image_temp

            if len(image_temp.shape) == 3:

                self.neuracrypt = NeuraCrypt(
                    image_size=(image_temp.shape[0] + pad_row, image_temp.shape[1] + pad_col, image_temp.shape[2]),
                    patch_size=int(self.block_size_dropdown_2.currentText()),
                    noise=noise,
                    noise_level=self.noise_slider.value())

            else:
                self.neuracrypt = NeuraCrypt(
                    image_size=(image_temp.shape[0] + pad_row, image_temp.shape[1] + pad_col),
                    patch_size=int(self.block_size_dropdown_2.currentText()),
                    noise=noise,
                    noise_level=self.noise_slider.value())

            # Encrypting the original image using the RMT encoder
            encrypted_img_array = self.neuracrypt.forward(image_padded).detach().numpy()

            if filePath:
                save_image_from_array(encrypted_img_array, filePath)
            # Convert the encoded image back to QPixmap and display on the label

            if len(encrypted_img_array.shape) == 3:  # RGB Image

                height, width, channel = encrypted_img_array.shape

                encrypted_img_bytes = bytes(encrypted_img_array)

                bytesPerLine = 3 * width

                qImg = QImage(encrypted_img_bytes, width, height, bytesPerLine,
                              QImage.Format_RGB888)

            else:  # Greyscale Image

                height, width = encrypted_img_array.shape

                bytesPerLine = width

                qImg = QImage(encrypted_img_array.data, width, height, bytesPerLine,
                              QImage.Format_Grayscale8)

            encrypted_pixmap = QPixmap.fromImage(qImg)

            self.lbl_disguised.setPixmap(encrypted_pixmap.scaled(200, 200))
            self.lbl_disguised_2.setPixmap(encrypted_pixmap.scaled(200, 200))

            self.chatbox_2.append("Encryption done!")

    def on_attack_click(self):

        image_files = self.image_files

        self.chatbox.append("Attacking: known pairs = " + str(self.knownpairs_slider.value()))

        pair = int(self.knownpairs_slider.value())

        rec = []

        index = random.sample(range(len(self.original_images)), pair)

        if len(self.original_images[0].shape) == 3:

            RMT_Mat = self.encoder.Estimate(np.array(self.original_images)[index, :, :, :],
                                                np.array(self.encrypted_images)[index, :, :, :])

        else:

            RMT_Mat = self.encoder.Estimate(np.array(self.original_images)[index, :, :],
                                                np.array(self.encrypted_images)[index, :, :])

        for i in range(len(self.encrypted_images)):
            encoded_img = self.encrypted_images[i]

            recover = self.encoder.Recover(encoded_img, RMT_Mat)

            rec.append(recover)

            print(np.linalg.norm(self.encoder.normalize(self.original_images[i]) - recover))

        self.chatbox.append("Please choose a place to store the recovered images")
        self.chatbox_2.append("Please choose a place to store the recovered images")

        options = QFileDialog.Options()

        recovered_images_save_path = QFileDialog.getExistingDirectory(self, "Choose Directory to Save recovered Images",
                                                                      "", options=options)

        for i in range(len(rec)):

            recovered_image = rec[i]

            recovered_image_path = os.path.join(recovered_images_save_path, "recovered_" + image_files[i])

            save_image_from_array(recovered_image, recovered_image_path)

            if i == self.index:

                if len(rec[i].shape) == 3:  # RGB Image

                    height, width, channel = recovered_image.shape

                    bytesPerLine = 3 * width

                    qImg = QImage(np.clip(rec[i], 0, 255).astype(np.uint8).data, width, height, bytesPerLine,
                                  QImage.Format_RGB888)

                else:  # Greyscale Image

                    height, width = recovered_image.shape

                    bytesPerLine = width

                    qImg = QImage(np.clip(rec[i], 0, 255).astype(np.uint8), width, height, bytesPerLine,
                                  QImage.Format_Grayscale8)

                recovered_pixmap = QPixmap.fromImage(qImg)

                self.lbl_attack_reconstructed.setPixmap(recovered_pixmap.scaled(200, 200))

        self.chatbox.append("Recovered!")
        self.chatbox_2.append("Recovered!")

    def update_knownpairs_label(self, value):

        self.knownpairs_label.setText("Number of known pairs: " + str(value))

    def update_noise_label(self, value):

        self.noise_slider_lbl.setText("Noise level: " + str(value))


# Application entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = ImageDisguisingApp()
    mainWin.show()
    sys.exit(app.exec_())
