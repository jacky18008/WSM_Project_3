import os 
from PIL import Image

class graph:
    #directory of the input image 
    input_dir = ''
    #directory of saving image 
    save_dir = ''

    #normal resize
    @classmethod
    def normal_resize(imga, width, height):
        img = Image.open(imga.input_dir)
        output = image.resize((weight, height), Image.ANTIALIAS)

    #cut resize
   @classmethod 
   def cut_and_resize(imga, width, height, cut_edge):
        img = Image.open(imga.input_dir)
        (x, y) = img.size
        if x > y and x > width:  
            region = (cut_edge, 0, x-cut_edge, y)  
            crop_img = img.crop(region)
            crop_img = crop_img.resize((width, height), Image.ANTIALIAS)
            crop_img.save(imga.save_dir)
        elif x < y and y > width:  
            region = (0, cut_edge, x, y-cut_edge)  
            crop_img = img.crop(region) 
            crop_img = crop_img.resize((width, height), Image.ANTIALIAS)
            crop_img.save(imga.save_dir)
        else:  
            crop_img = img.resize((width, height), Image.ANTIALIAS)  
            crop_img.save(imga.save_dir)  
            
    #cut into square and resize
    @classmethod 
    def cut_and_resize_in_squre(imga, width, height):
        img = Image.open(imga.input_dir)
        (x, y) = img.size
        if x > y and x > width:
            region = ((x-y)/2, 0, y+(x-y)/2, y)  
            crop_img = img.crop(region)
            crop_img = crop_img.resize((width, height), Image.ANTIALIAS)
            crop_img.save(imga.save_dir)
        elif x < y and y > width:  
            region = (0, (y-x)/2, x, x+(y-x)/2)  
            crop_img = img.crop(region) 
            crop_img = crop_img.resize((width, height), Image.ANTIALIAS)
            crop_img.save(imga.save_dir)
        else:  
            crop_img = img.resize((width, height), Image.ANTIALIAS)  
            crop_img.save(imga.save_dir)
    
    
