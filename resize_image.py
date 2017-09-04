from PIL import Image
import os
import sys

list = os.listdir(sys.argv[1])
print(list)

for image in list:
    id_tag = image.find(".")
    name=image[0:id_tag]
    print(name)

    im=Image.open(sys.argv[1]+image)
    if (im.size[0] > im.size[1]):
        proportion = im.size[1]/256 +1
    else:
        proportion = im.size[0]/256 +1
    out = im.resize((im.size[0]/proportion, im.size[1]/proportion))
    #out.show()
    out.save(sys.argv[2]+name+".jpg")

