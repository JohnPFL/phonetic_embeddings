from visually_grounded_embs.visually_grounded_embs import VGE
from visually_grounded_embs.embeddings_analysis import ec_loader
from PIL import Image, ImageFont
from os.path import join
import os
os.chdir('/home/sperduti/vgm/')

font_path = 'vge_utils/font'
font_alt_path = 'vge_utils/font_alternatives'
font = 'NotoSans-Regular.ttf'
font_images = 'vge_utils/normalized_images'
size = 16
vge = VGE(join(font_path, font), font_alt_path, font_images, '/home/sperduti/vgm/vge_utils/missing_char_path', 10, size, 15)
    
def print_letters(letter_list, img_width, img_height, output_path):
    # Create a blank image
    output_image = Image.new("L", (img_width * len(letter_list), img_height), color=255)
    x_offset = 0
    for letter in letter_list:
        # Create the letter image
        letter_image = vge.image_printer(letter)
        # Paste the letter image onto the output image
        output_image.paste(letter_image, (x_offset, 0))
        x_offset += img_width
    # Return the output image
    output_image.save(output_path)

if __name__ == '__main__':
    txt_file_to_print = '/home/sperduti/vgm/Datasets/misspellings.txt'
    font_path = 'vge_utils/font'
    image_path = 'vge_utils/control_ec_images/final_image.png'
    img_width = 10
    img_height = 16
    font_size = 15

    letter_list = ec_loader(txt_file_to_print)
    #flattening the list of characters
    flat_letter_list = [item for sublist in letter_list for item in sublist]
    print_letters(flat_letter_list, img_width, img_height, image_path)


    