from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from imutils.video import VideoStream
from imutils.video import FPS
import time
from tkinter import *
from tkinter import messagebox
import pandas as pd

ARIAL = ("Arial", 12, "bold")

class BankUi:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart ATM Machine")
        self.root.geometry("800x500")
        self.root.configure(bg="#E0F7FA")  # Light blue background color

        self.countter = 2  # For retry count in face verification
        self.real_user = None

        self.header = Label(self.root, text="MULTI BANK", bg="#7B1FA2", fg="white", font=("Arial", 24, "bold"))
        self.header.pack(fill=X, pady=10)

        self.frame = Frame(self.root, bg="#BA68C8", width=900, height=500, bd=5, relief="groove")  # Added border
        self.frame.pack(pady=20)

        self.button1 = Button(self.frame, text="Click to begin transactions", bg="#FF4081", fg="white", font=ARIAL,
                              command=self.begin_page, width=20)
        self.button1.pack(pady=20, padx=20, fill=X)

        self.q = Button(self.frame, text="Quit", bg="#E53935", fg="white", font=ARIAL, command=self.root.destroy, width=10)
        self.q.pack(pady=10)

    def begin_page(self):
        self.frame.destroy()
        self.frame = Frame(self.root, bg="#BA68C8", bd=5, relief="groove")  # Added border
        self.frame.pack(pady=20, fill=BOTH, expand=True)

        self.enroll = Button(self.frame, text="Enroll", bg="#FF4081", fg="white", font=ARIAL, command=self.enroll_user, width=15)
        self.withdraw = Button(self.frame, text="Login to Banking Transactions", bg="blue", fg="white", font=ARIAL,  # Changed color to blue
                               command=self.withdraw_money_page, width=15)
        self.q = Button(self.frame, text="Quit", bg="#E53935", fg="white", font=ARIAL, command=self.root.destroy, width=10)

        self.enroll.pack(pady=20, padx=40, fill=X)
        self.withdraw.pack(pady=10, padx=40, fill=X)
        self.q.pack(pady=10, padx=40)

    def withdraw_money_page(self):
        self.frame.destroy()
        self.frame = Frame(self.root, bg="#BA68C8", bd=5, relief="groove")  # Added border
        self.frame.pack(pady=20, fill=BOTH, expand=True)

        instructions = [
            "1. Click 'Verify Face Id' to perform facial recognition.",
            "2. Each capture will take 15 seconds; move your face in different directions.",
            "3. If recognized, input your account password.",
            "4. If not recognized after 5 seconds, you have 2 more trials.",
            "5. If not recognized after three trials, you cannot withdraw.",
            "6. Click 'Verify Face Id' to begin."
        ]

        for instruction in instructions:
            label = Label(self.frame, text=instruction, bg="#BA68C8", fg="white", font=ARIAL, anchor="w")
            label.pack(pady=3, padx=30, fill=X)

        self.button = Button(self.frame, text="Verify Face Id", bg="green", fg="white", font=ARIAL,  # Changed color to green
                             command=self.video_check, width=20)
        self.button.pack(pady=20, padx=100, fill=X)

        btn_frame = Frame(self.frame, bg="#BA68C8")
        btn_frame.pack(pady=10)

        self.b = Button(btn_frame, text="Back", bg="#FF4081", fg="white", font=ARIAL, command=self.begin_page, width=10)
        self.b.pack(side=LEFT, padx=20)
        self.q = Button(btn_frame, text="Quit", bg="#E53935", fg="white", font=ARIAL, command=self.root.destroy, width=10)
        self.q.pack(side=LEFT, padx=20)

    def enroll_user(self):
        self.frame.destroy()
        self.frame = Frame(self.root, bg="#BA68C8", bd=5, relief="groove")  # Added border
        self.frame.pack(pady=20, fill=BOTH, expand=True)

        self.userlabel = Label(self.frame, text="Full Name", bg="#BA68C8", fg="white", font=ARIAL, anchor="w")
        self.userlabel.pack(pady=(20,5), padx=50, fill=X)
        self.uentry = Entry(self.frame, bg="honeydew", highlightcolor="#FF4081",
                            highlightthickness=2, highlightbackground="white")
        self.uentry.pack(pady=5, padx=50, fill=X)

        self.plabel = Label(self.frame, text="Password", bg="#BA68C8", fg="white", font=ARIAL, anchor="w")
        self.plabel.pack(pady=(20,5), padx=50, fill=X)
        self.pentry = Entry(self.frame, bg="honeydew", show="*", highlightcolor="#FF4081",
                            highlightthickness=2, highlightbackground="white")
        self.pentry.pack(pady=5, padx=50, fill=X)

        button_frame = Frame(self.frame, bg="#BA68C8")
        button_frame.pack(pady=20)

        self.button1 = Button(button_frame, text="Next", bg="#FF4081", fg="white", font=ARIAL,
                              command=self.enroll_and_move_to_next_screen, width=10)
        self.button1.pack(side=LEFT, padx=10)

        self.b = Button(button_frame, text="Back", bg="#FF4081", fg="white", font=ARIAL, command=self.begin_page, width=10)
        self.b.pack(side=LEFT, padx=10)

        self.q = Button(button_frame, text="Quit", bg="#E53935", fg="white", font=ARIAL, command=self.root.destroy, width=10)
        self.q.pack(side=LEFT, padx=10)

    def enroll_and_move_to_next_screen(self):
        name = self.uentry.get()
        password = self.pentry.get()
        if not name and not password:
            messagebox._show("Error", "You need a name to enroll an account and you need to input a password!")
            self.enroll_user()
        elif not password:
            messagebox._show("Error", "You need to input a password!")
            self.enroll_user()
        elif not name:
            messagebox._show("Error", "You need a name to enroll an account!")
            self.enroll_user()
        elif len(password) < 8:
            messagebox._show("Password Error", "Your password needs to be at least 8 digits!")
            self.enroll_user()
        else:
            self.write_to_csv()
            self.video_capture_page()

    def password_verification(self):
        self.frame.destroy()
        self.frame = Frame(self.root, bg="#BA68C8", bd=5, relief="groove")  # Added border
        self.frame.pack(pady=20, fill=BOTH, expand=True)

        self.plabel = Label(self.frame, text="Please enter your account password", bg="#BA68C8", fg="white", font=ARIAL)
        self.plabel.pack(pady=(20,5), padx=50, fill=X)

        self.givenpentry = Entry(self.frame, bg="honeydew", show="*", highlightcolor="#FF4081",
                                highlightthickness=2, highlightbackground="white")
        self.givenpentry.pack(pady=5, padx=50, fill=X)

        button_frame = Frame(self.frame, bg="#BA68C8")
        button_frame.pack(pady=20)

        self.button1 = Button(button_frame, text="Verify", bg="green", fg="white", font=ARIAL, command=self.verify_user, width=10)  # Changed color to green
        self.button1.pack(side=LEFT, padx=10)

        self.b = Button(button_frame, text="Back", bg="#FF4081", fg="white", font=ARIAL, command=self.begin_page, width=10)
        self.b.pack(side=LEFT, padx=10)

        self.q = Button(button_frame, text="Quit", bg="#E53935", fg="white", font=ARIAL, command=self.root.destroy, width=10)
        self.q.pack(side=LEFT, padx=10)

    def verify_user(self):
        data = pd.read_csv('bank_details.csv')
        self.gottenpassword = data[data.loc[:, 'unique_id'] == self.real_user].loc[:, 'password'].values[0]
        if str(self.givenpentry.get()) == str(self.gottenpassword):
            messagebox._show("Verification Info!", "Verification Successful!")
            self.final_page()
        else:
            messagebox._show("Verification Info!", "Verification Failed")
            self.begin_page()

    def final_page(self):
        self.frame.destroy()
        self.frame = Frame(self.root, bg="#BA68C8", bd=5, relief="groove")  # Added border
        self.frame.pack(pady=20, fill=BOTH, expand=True)

        btn_frame = Frame(self.frame, bg="#BA68C8")
        btn_frame.pack(pady=10, fill=X, padx=50)

        self.detail = Button(btn_frame, text="Transfer", bg="#FF4081", fg="white", font=ARIAL,
                             command=self.user_account_transfer, width=8)  # Reduced width
        self.detail.pack(pady=10, fill=X)

        self.enquiry = Button(btn_frame, text="Balance Enquiry", bg="yellow", fg="black", font=ARIAL,
                             command=self.user_balance, width=8)  # Reduced width
        self.enquiry.pack(pady=10, fill=X)

        self.deposit = Button(btn_frame, text="Deposit Money", bg="green", fg="white", font=ARIAL,
                              command=self.user_deposit_money, width=8)  # Reduced width
        self.deposit.pack(pady=10, fill=X)

        self.withdrawl = Button(btn_frame, text="Withdraw Money", bg="blue", fg="white", font=ARIAL,
                                command=self.user_withdrawl_money, width=8)  # Reduced width and fixed typo Withdrawl -> Withdraw
        self.withdrawl.pack(pady=10, fill=X)

        self.q = Button(btn_frame, text="Log out", bg="#E53935", fg="white", font=ARIAL, command=self.begin_page, width=8)  # Reduced width
        self.q.pack(pady=10, fill=X)

    # Other methods remain the same with original widths for buttons outside final_page
    # (or reduced if you want to change them similarly; your request was only to reduce final page buttons)
    # For brevity, including rest unchanged methods below:

    def user_account_transfer(self):
        self.frame.destroy()
        self.frame = Frame(self.root, bg="#BA68C8", bd=5, relief="groove")
        self.frame.pack(pady=20, fill=BOTH, expand=True)

        btn_frame = Frame(self.frame, bg="#BA68C8")
        btn_frame.pack(pady=5, fill=X, padx=50)

        self.detail = Button(btn_frame, text="Transfer", bg="#FF4081", fg="white", font=ARIAL,
                             command=self.user_account_transfer, width=10)
        self.detail.pack(pady=5, fill=X)

        self.enquiry = Button(btn_frame, text="Balance Enquiry", bg="yellow", fg="black", font=ARIAL,
                              command=self.user_balance, width=10)
        self.enquiry.pack(pady=5, fill=X)

        self.deposit = Button(btn_frame, text="Deposit Money", bg="green", fg="white", font=ARIAL,
                              command=self.user_deposit_money, width=10)
        self.deposit.pack(pady=5, fill=X)

        self.withdrawl = Button(btn_frame, text="Withdraw Money", bg="blue", fg="white", font=ARIAL,
                                command=self.user_withdrawl_money, width=10)
        self.withdrawl.pack(pady=5, fill=X)

        self.q = Button(btn_frame, text="Log out", bg="#E53935", fg="white", font=ARIAL, command=self.begin_page, width=10)
        self.q.pack(pady=10, fill=X)

        lbl_frame = Frame(self.frame, bg="#BA68C8")
        lbl_frame.pack(pady=20, fill=X, padx=50)

        self.label11 = Label(lbl_frame, text="Please enter the recipient's account number", bg="#BA68C8", fg="white",
                             font=ARIAL, anchor="w")
        self.label11.pack(pady=5, fill=X)
        self.entry11 = Entry(lbl_frame, bg="honeydew", highlightcolor="#FF4081", highlightthickness=2,
                             highlightbackground="white")
        self.entry11.pack(pady=5, fill=X)

        self.label21 = Label(lbl_frame, text="Please enter the amount to be transferred", bg="#BA68C8", fg="white",
                             font=ARIAL, anchor="w")
        self.label21.pack(pady=5, fill=X)
        self.entry21 = Entry(lbl_frame, bg="honeydew", highlightcolor="#FF4081", highlightthickness=2,
                             highlightbackground="white")
        self.entry21.pack(pady=5, fill=X)

        self.button1 = Button(lbl_frame, text="Transfer", bg="#FF4081", fg="white", font=ARIAL,
                              command=self.user_account_transfer_transc, width=15)
        self.button1.pack(pady=10)

    def user_account_transfer_transc(self):
        data = pd.read_csv('bank_details.csv')
        try:
            acc_num = int(self.entry11.get())
            amount = int(self.entry21.get())
        except ValueError:
            messagebox._show("Transfer Info!", "Please enter valid numeric values!")
            return

        if acc_num not in data['account_number'].values:
            messagebox._show("Transfer Info!", "Invalid account number")
        elif acc_num == data[data['unique_id'] == self.real_user]['account_number'].values[0]:
            messagebox._show("Transfer Info!", "Sorry, you cannot make a transfer to yourself")
        elif amount > data[data['unique_id'] == self.real_user]['account_balance'].values[0]:
            messagebox._show("Transfer Info!", "Insufficient Funds")
        else:
            update_data = data.set_index('account_number')
            update_data.loc[acc_num, 'account_balance'] += amount
            update_data.loc[data[data['unique_id'] == self.real_user]['account_number'].values[0], 'account_balance'] -= amount
            update_data['account_number'] = update_data.index
            update_data.reset_index(drop=True, inplace=True)
            update_data = update_data.reindex(columns=['unique_id', 'account_number', 'name', 'bank', 'password', 'account_balance'])
            update_data.to_csv('bank_details.csv', index=False)
            messagebox._show("Transfer Info!", "Successfully Transferred")

    def user_balance(self):
        self.frame.destroy()
        self.frame = Frame(self.root, bg="#BA68C8", bd=5, relief="groove")
        self.frame.pack(pady=20, fill=BOTH, expand=True)

        btn_frame = Frame(self.frame, bg="#BA68C8")
        btn_frame.pack(pady=10, fill=X, padx=50)

        self.detail = Button(btn_frame, text="Transfer", bg="#FF4081", fg="white", font=ARIAL,
                             command=self.user_account_transfer, width=10)
        self.detail.pack(pady=10, fill=X)

        self.enquiry = Button(btn_frame, text="Balance Enquiry", bg="yellow", fg="black", font=ARIAL,
                              command=self.user_balance, width=10)
        self.enquiry.pack(pady=10, fill=X)

        self.deposit = Button(btn_frame, text="Deposit Money", bg="green", fg="white", font=ARIAL,
                              command=self.user_deposit_money, width=10)
        self.deposit.pack(pady=10, fill=X)

        self.withdrawl = Button(btn_frame, text="Withdraw Money", bg="blue", fg="white", font=ARIAL,
                                command=self.user_withdrawl_money, width=10)
        self.withdrawl.pack(pady=10, fill=X)

        self.q = Button(btn_frame, text="Log out", bg="#E53935", fg="white", font=ARIAL, command=self.begin_page, width=10)
        self.q.pack(pady=10, fill=X)

        data = pd.read_csv('bank_details.csv')
        balance = data[data['unique_id'] == self.real_user]['account_balance'].values[0]

        self.label = Label(self.frame, text='Current Account Balance: N{}'.format(balance), font=ARIAL, bg="#BA68C8", fg="white")
        self.label.pack(pady=30)

    def user_deposit_money(self):
        self.frame.destroy()
        self.frame = Frame(self.root, bg="#BA68C8", bd=5, relief="groove")
        self.frame.pack(pady=20, fill=BOTH, expand=True)

        btn_frame = Frame(self.frame, bg="#BA68C8")
        btn_frame.pack(pady=10, fill=X, padx=50)

        self.detail = Button(btn_frame, text="Transfer", bg="#FF4081", fg="white", font=ARIAL,
                             command=self.user_account_transfer, width=10)
        self.detail.pack(pady=10, fill=X)

        self.enquiry = Button(btn_frame, text="Balance Enquiry", bg="yellow", fg="black", font=ARIAL,
                              command=self.user_balance, width=10)
        self.enquiry.pack(pady=10, fill=X)

        self.deposit = Button(btn_frame, text="Deposit Money", bg="green", fg="white", font=ARIAL,
                              command=self.user_deposit_money, width=10)
        self.deposit.pack(pady=10, fill=X)

        self.withdrawl = Button(btn_frame, text="Withdraw Money", bg="blue", fg="white", font=ARIAL,
                                command=self.user_withdrawl_money, width=10)
        self.withdrawl.pack(pady=10, fill=X)

        self.q = Button(btn_frame, text="Log out", bg="#E53935", fg="white", font=ARIAL, command=self.begin_page, width=10)
        self.q.pack(pady=10, fill=X)

        self.label = Label(self.frame, text="Enter amount", font=ARIAL, bg="#BA68C8", fg="white")
        self.label.pack(pady=20)

        self.money_box = Entry(self.frame, bg="honeydew", highlightcolor="#FF4081", highlightthickness=2, highlightbackground="white")
        self.money_box.pack(pady=10, padx=50, fill=X)

        self.submitButton = Button(self.frame, text="Deposit", bg="#FF4081", fg="white", font=ARIAL,
                                   command=lambda e=None: self.user_deposit_trans(e), width=15)
        self.submitButton.pack(pady=10)

    def user_deposit_trans(self, flag):
        try:
            amount = int(self.money_box.get())
        except ValueError:
            messagebox._show("Deposit Info!", "Please enter a valid amount!")
            return

        data = pd.read_csv('bank_details.csv')
        update_data = data.set_index('unique_id')
        update_data.loc[self.real_user, 'account_balance'] += amount
        update_data.reset_index(inplace=True)
        update_data.columns = ['unique_id', 'account_number', 'name', 'bank', 'password', 'account_balance']
        update_data.to_csv('bank_details.csv', index=False)
        messagebox._show("Deposit Info!", "Successfully Deposited!")

    def user_withdrawl_money(self):
        self.frame.destroy()
        self.frame = Frame(self.root, bg="#BA68C8", bd=5, relief="groove")
        self.frame.pack(pady=20, fill=BOTH, expand=True)

        btn_frame = Frame(self.frame, bg="#BA68C8")
        btn_frame.pack(pady=10, fill=X, padx=50)

        self.detail = Button(btn_frame, text="Transfer", bg="#FF4081", fg="white", font=ARIAL,
                             command=self.user_account_transfer, width=10)
        self.detail.pack(pady=10, fill=X)

        self.enquiry = Button(btn_frame, text="Balance Enquiry", bg="yellow", fg="black", font=ARIAL,
                              command=self.user_balance, width=10)
        self.enquiry.pack(pady=10, fill=X)

        self.deposit = Button(btn_frame, text="Deposit Money", bg="green", fg="white", font=ARIAL,
                              command=self.user_deposit_money, width=10)
        self.deposit.pack(pady=10, fill=X)

        self.withdrawl = Button(btn_frame, text="Withdraw Money", bg="blue", fg="white", font=ARIAL,
                                command=self.user_withdrawl_money, width=10)
        self.withdrawl.pack(pady=10, fill=X)

        self.q = Button(btn_frame, text="Log out", bg="#E53935", fg="white", font=ARIAL, command=self.begin_page, width=10)
        self.q.pack(pady=10, fill=X)

        self.label = Label(self.frame, text="Enter amount", font=ARIAL, bg="#BA68C8", fg="white")
        self.label.pack(pady=20)

        self.money_box = Entry(self.frame, bg="honeydew", highlightcolor="#FF4081", highlightthickness=2, highlightbackground="white")
        self.money_box.pack(pady=10, padx=50, fill=X)

        self.submitButton = Button(self.frame, text="Withdraw", bg="blue", fg="white", font=ARIAL,
                                   command=lambda e=None: self.user_withdrawl_trans(e), width=15)
        self.submitButton.pack(pady=10)

    def user_withdrawl_trans(self, flag):
        try:
            amount = int(self.money_box.get())
        except ValueError:
            messagebox._show("Withdrawal Info!", "Please enter a valid amount!")
            return

        data = pd.read_csv('bank_details.csv')
        update_data = data.set_index('unique_id')
        if amount <= update_data.loc[self.real_user, 'account_balance']:
            update_data.loc[self.real_user, 'account_balance'] -= amount
            update_data.reset_index(inplace=True)
            update_data.columns = ['unique_id', 'account_number', 'name', 'bank', 'password', 'account_balance']
            update_data.to_csv('bank_details.csv', index=False)
            messagebox._show("Withdrawal Info!", "Successfully Withdrawn, please take your cash")
        else:
            messagebox._show("Withdrawal Info!", "Insufficient Funds")

    def write_to_csv(self):
        import csv
        from random import randint
        n = 10
        range_start = 10 ** (n - 1)
        range_end = (10 ** n) - 1
        account_number = randint(range_start, range_end)
        n = 5
        range_start = 10 ** (n - 1)
        range_end = (10 ** n) - 1
        unique_id = randint(range_start, range_end)
        bank = "Unilag Bank"
        account_balance = "10000"
        name = self.uentry.get()
        password = self.pentry.get()
        with open(r'bank_details.csv', 'a', newline='\n') as f:
            writer = csv.writer(f)
            writer.writerow([unique_id, account_number, name, bank, password, account_balance])
        messagebox._show("Enrollment Info!", "Successfully Enrolled!")

    def video_capture_page(self):
        self.frame.destroy()
        self.frame = Frame(self.root, bg="#BA68C8", bd=5, relief="groove")
        self.frame.pack(pady=20, fill=BOTH, expand=True)

        instructions = [
            "1. Click 'Capture' button to capture your image.",
            "2. Capture 5 images for full registration.",
            "3. Press space bar on your keyboard to capture each image.",
            "4. Wait for notification before leaving the page.",
            "5. Click 'Capture' button and press space for new image."
        ]

        for instruction in instructions:
            label = Label(self.frame, text=instruction, bg="#BA68C8", fg="white", font=ARIAL, anchor="w")
            label.pack(pady=3, padx=30, fill=X)

        self.button = Button(self.frame, text="Capture", bg="#FF4081", fg="white", font=ARIAL,
                             command=self.captureuser, width=20)
        self.button.pack(pady=20, padx=100, fill=X)

    def captureuser(self):
        data = pd.read_csv('bank_details.csv')
        name = data.loc[:, 'unique_id'].values[-1]
        cam = cv2.VideoCapture(0)

        cv2.namedWindow("capture")

        img_counter = 0

        dirname = f'dataset/{name}'
        try:
            os.mkdir(dirname)
        except FileExistsError:
            pass

        while True:
            ret, frame = cam.read()
            cv2.imshow("capture", frame)

            if img_counter == 5:
                cv2.destroyWindow("capture")
                break
            if not ret:
                break
            k = cv2.waitKey(1)

            if k % 256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k % 256 == 32:
                # SPACE pressed
                path = f'dataset/{name}'
                img_name = "{}.jpg".format(img_counter)
                cv2.imwrite(os.path.join(path, img_name), frame)
                print("{} written!".format(img_name))
                img_counter += 1

        cam.release()

        cv2.destroyAllWindows()

        self.get_embeddings()
        self.train_model()
        messagebox._show('Registration Info!', "Face Id Successfully Registered!")
        self.begin_page()

    def get_embeddings(self):
        print("[INFO] loading face detector...")

        detector = cv2.dnn.readNetFromCaffe('face_detection_model/deploy.prototxt',
                                            'face_detection_model/res10_300x300_ssd_iter_140000.caffemodel')
        embedder = cv2.dnn.readNetFromTorch('nn4.small2.v1.t7')

        print("[INFO] quantifying faces...")
        imagePaths = list(paths.list_images('dataset'))

        knownEmbeddings = []
        knownNames = []

        total = 0

        for (i, imagePath) in enumerate(imagePaths):
            print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
            name = imagePath.split(os.path.sep)[-2]

            image = cv2.imread(imagePath)
            image = imutils.resize(image, width=600)
            (h, w) = image.shape[:2]
            imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

            detector.setInput(imageBlob)
            detections = detector.forward()

            if len(detections) > 0:
                i = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, i, 2]

                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    face = image[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]

                    if fW < 20 or fH < 20:
                        continue

                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()

                    knownNames.append(name)
                    knownEmbeddings.append(vec.flatten())
                    total += 1

        print("[INFO] serializing {} encodings...".format(total))
        data = {"embeddings": knownEmbeddings, "names": knownNames}
        with open('output/embeddings.pickle', "wb") as f:
            f.write(pickle.dumps(data))

    def train_model(self):
        print("[INFO] loading face embeddings...")
        data = pickle.loads(open('output/embeddings.pickle', "rb").read())
        le = LabelEncoder()
        labels = le.fit_transform(data["names"])

        print("[INFO] training model...")
        recognizer = SVC(C=1.0, kernel="linear", probability=True)
        recognizer.fit(data["embeddings"], labels)

        with open('output/recognizer.pickle', "wb") as f:
            f.write(pickle.dumps(recognizer))

        with open('output/le.pickle', "wb") as f:
            f.write(pickle.dumps(le))

    def video_check(self):
        detector = cv2.dnn.readNetFromCaffe('face_detection_model/deploy.prototxt',
                                            'face_detection_model/res10_300x300_ssd_iter_140000.caffemodel')

        print("[INFO] loading face recognizer...")
        embedder = cv2.dnn.readNetFromTorch('nn4.small2.v1.t7')

        recognizer = pickle.loads(open('output/recognizer.pickle', "rb").read())
        le = pickle.loads(open('output/le.pickle', "rb").read())

        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(2.0)

        timeout = time.time() + 5

        fps = FPS().start()

        real_user_list = []

        while True:
            if time.time() > timeout:
                cv2.destroyWindow("Frame")
                break

            frame = vs.read()
            frame = imutils.resize(frame, width=800)
            (h, w) = frame.shape[:2]

            imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

            detector.setInput(imageBlob)
            detections = detector.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    face = frame[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]

                    if fW < 20 or fH < 20:
                        continue

                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()

                    preds = recognizer.predict_proba(vec)[0]
                    j = np.argmax(preds)
                    proba = preds[j]
                    name = le.classes_[j]

                    if (name == 'unknown') or (proba * 100) < 50:
                        real_user_list.append(name)
                    else:
                        real_user_list.append(name)
                        break

            fps.update()

            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        cv2.destroyAllWindows()
        vs.stop()
        print(real_user_list)

        try:
            unknown_check = Counter(real_user_list).most_common(1)[0][0] == 'unknown'
        except IndexError:
            unknown_check = True

        if unknown_check:
            if self.countter > 0:
                messagebox._show("Verification Info!", "Face Id match failed! You have {} trials left".format(self.countter))
                self.countter -= 1
                self.video_check()
            else:
                messagebox._show("Verification Info!", "Face Id match failed! You cannot withdraw at this time, try again later")
                self.begin_page()
                self.countter = 2
        else:
            self.real_user = int(Counter(real_user_list).most_common(1)[0][0])
            messagebox._show("Verification Info!", "Face Id match!")
            self.password_verification()

if __name__ == "__main__":
    root = Tk()
    obj = BankUi(root)
    root.mainloop()
