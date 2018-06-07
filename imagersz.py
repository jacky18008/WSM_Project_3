import os 
from PIL import image

class graph:
    #directory of the input image 
    input_dir = ''
    #directory of saving image 
    save_dir = ''

    #normal resize
    @classmethod
    def normal_resize(imga, width, height):
        img = image.open(imga.input_dir)
        output = image.resize((weight, height), image.ANTIALIAS)

    #cut and resize
    @classmethod 
    def cut_and_resize(imga, width, height, cut_edge):
        img = image.open(imga.input_dir)
        (x, y) = img.size
        if x > y && x > width:  
            region = (cut_edge, 0, x-cut_edge, y)  
            crop_img = img.crop(region)
            crop_img = crop_img.resize((width, height), image.ANTIALIAS)
            crop_img.save(imga.save_dir)
        elif x < y && y > width:  
            region = (0, cut_edge, x, y-cut_edge)  
            crop_img = img.crop(region) 
            crop_img = crop_img.resize((width, height), image.ANTIALIAS)
            crop_img.save(imga.save_dir)
        else:  
            crop_img = img.resize((width, height), image.ANTIALIAS)  
            crop_img.save(imga.save_dir)  
            
    #cut into square and resize
    @classmethod 
    def cut_and_resize_in_squre(imga, width, height):
        img = image.open(imga.input_dir)
        (x, y) = img.size
        if x > y && x > width:
            region = ((x-y)/2, 0, y+(x-y)/2, y)  
            crop_img = img.crop(region)
            crop_img = crop_img.resize((width, height), image.ANTIALIAS)
            crop_img.save(imga.save_dir)
        elif x < y && y > width:  
            region = (0, (y-x)/2, x, x+(y-x)/2)  
            crop_img = img.crop(region) 
            crop_img = crop_img.resize((width, height), image.ANTIALIAS)
            crop_img.save(imga.save_dir)
        else:  
            crop_img = img.resize((width, height), image.ANTIALIAS)  
            crop_img.save(imga.save_dir)
    
    
