from tkinter import Tk
from database import initialize_db  # Adjust this if necessary
from gui import ProctoringApp

def main():
    initialize_db()
    root = Tk()
    app = ProctoringApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()