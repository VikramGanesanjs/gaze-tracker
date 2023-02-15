import tkinter as tk
import cv2
import threading
from gazer import Gazer
import PIL.Image, PIL.ImageTk

class EyeTyper(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("{0}x{1}+0+0".format(self.winfo_screenwidth(), self.winfo_screenheight()))
        self.pages = [["A", "B", "C", "D", "E","F",], ["G", "H", "I", "J","K", "L",], [ "M", "N", "O","P", "Q", "R",], ["S", "T", "U", "V", "W", "X",], ["Y","Z", "Space", "1", "2", "3"], ["4", "5", "6", "7", "8", "9",], [ "0", "!", "-", "Delete", "@", "$",]]
        self.current_page = 0
        self.buttons = []
        self.gazer = Gazer()
        self.vid = cv2.VideoCapture(0)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(self, width = 500, height=200)
        self.canvas.grid(row=3, column=0, columnspan=7, sticky="nsew")
        left_arrow = tk.Button(self, text="<", font=("Helvetica", 48), command=self.left_arrow)
        left_arrow.grid(column= 0, sticky="nsew")
        for letter in self.pages[self.current_page]:
            on_letter_click = lambda x = letter: self.on_letter_click(x)
            if letter == "Space":
                on_letter_click = lambda x = letter: self.on_letter_click(" ")
                self.buttons.append(tk.Button(self, text=letter, font=("Helvetica", 48), command=on_letter_click))
            elif letter == "Delete":
                self.buttons.append(tk.Button(self, text=letter, font=("Helvetica", 48), command=self.delete_letter))
            else:
               self.buttons.append(tk.Button(self, text=letter, font=("Helvetica", 48), command=on_letter_click)) 
        for i, button in enumerate(self.buttons):
            button.grid(row=int(i/3), column=i+1, sticky="nsew")
        right_arrow = tk.Button(self, text=">", font=("Helvetica", 48), command=self.right_arrow)
        right_arrow.grid(column= 6, sticky="nsew")
        self.word = ""

        label = tk.Label(self, text="Text: " + self.word, font=("Helvetica", 24))
        label.grid(row=2, column=0, columnspan=7, sticky="nsew")

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.columnconfigure(3, weight=1)
        self.columnconfigure(4, weight=1)
        self.columnconfigure(5, weight=1)
        self.columnconfigure(6, weight=1)
        self.rowconfigure(0, weight=3)
        self.rowconfigure(1, weight=3)
        self.rowconfigure(2, weight=2)
        self.rowconfigure(3, weight=2)
        self.delay = 15
        self.update()


    def on_letter_click(self, letter):
        self.word += letter
        print(self.word)
        self.update_page()
    
    def delete_letter(self):
        self.word = self.word[:-1]
        self.update_page()

    def clear_screen(self):
        for widget in self.winfo_children():
            widget.destroy()

    def update(self):
        # Get a frame from the video source
        self.update_page()
        ret, frame = self.vid.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv2.flip(frame, 1)).resize((500, 200)))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
            x, y = self.gazer.without_video_cap(frame)
            self.gazer.move_mouse(x, y, 0.01)
        self.after(self.delay, self.update)
        

    def update_page(self):
        self.clear_screen()
        self.canvas = tk.Canvas(self, width = 500, 
                                height = 200)
        self.canvas.grid(row=3, column=0, columnspan=7, sticky="nsew")
        left_arrow = tk.Button(self, text="<", font=("Helvetica", 48), command=self.left_arrow)
        left_arrow.grid(column= 0, sticky="nsew")
        self.buttons = []
        for letter in self.pages[self.current_page]:
            on_letter_click = lambda x = letter: self.on_letter_click(x)
            if letter == "Space":
                on_letter_click = lambda x = letter: self.on_letter_click(" ")
                self.buttons.append(tk.Button(self, text=letter, font=("Helvetica", 48), command=on_letter_click))
            elif letter == "Delete":
                self.buttons.append(tk.Button(self, text=letter, font=("Helvetica", 48), command=self.delete_letter))
            else:
               self.buttons.append(tk.Button(self, text=letter, font=("Helvetica", 48), command=on_letter_click)) 
        for i, button in enumerate(self.buttons):
            button.grid(row=int(i/3), column=i+1, sticky="nsew")
        right_arrow = tk.Button(self, text=">", font=("Helvetica", 48), command=self.right_arrow)
        right_arrow.grid(column= 6, sticky="nsew")
        label = tk.Label(self, text="Text: " + self.word, font=("Helvetica", 24))
        label.grid(row=2, column=0, columnspan=7, sticky="nsew")

    def left_arrow(self):
        self.current_page = (self.current_page - 1) % len(self.pages)
        self.update_page()

    def right_arrow(self):
        self.current_page = (self.current_page + 1) % len(self.pages)
        self.update_page()

if __name__ == "__main__":
    window = EyeTyper()
    window.mainloop()
