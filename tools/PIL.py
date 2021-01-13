
#Import required library
from PIL import Image

#Open Image
im = Image.open("TajMahal.jpg")

#Image rotate & show
im.rotate(45).show()
