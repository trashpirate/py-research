

import os

# https://www.tutorialspoint.com/How-can-I-create-a-directory-if-it-does-not-exist-using-Python
def createDirectory(path):

    if not os.path.exists(path):
        os.makedirs(path)
        print("Created directory at path: ", os.path.abspath(path))