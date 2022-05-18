import tkinter as tk
from FUSION.interface.Application import Application
from FUSION.interface.Application import Application
import pathlib
pathlib.Path(__file__).parent.resolve()
# def main(tk.Tk):
#     ############################################################################
#     # definition of the window
#     root = tk.Tk()
#     app = Application(root)
#     ###########################################################################
#     # Implementation of the canvas
#     # screen(app)
#     # menu(app)
#     app.mainloop()


#
# if __name__ == "__main__":
#     main()

app = Application(master=tk.Tk())
app.mainloop()
