from PIL import Image, ImageDraw, ImageFont
import os

curdir = os.path.abspath(os.path.dirname(__file__))

def generate_letters(min_width, min_height, output_folder=os.path.join(curdir, "res", "letters")):
    os.makedirs(output_folder, exist_ok=True)

    characters = [chr(x) for x in range(ord('A'), ord('Z') + 1)] + [str(x) for x in range(0, 10)]

    font_size = min(min_width, min_height) * 0.8
     
    font_path = os.path.join(curdir, "res", "ArielBold.ttf")
    font = ImageFont.truetype(font_path, font_size)
    for char in characters:
        bounds = font.getbbox(char)
        letter_width, letter_height = bounds[2] - bounds[0], bounds[3] - bounds[1]

        image = Image.new("RGB", (min_width, min_height), "White")
        draw = ImageDraw.Draw(image)

        x = (min_width - letter_width) // 2
        y = (min_height - letter_height) // 2
        draw.text((x, y), char, font=font, fill="black")

        image.save(f"{output_folder}/{char}.png", "PNG")

if __name__ == "__main__":
    generate_letters(320, 320)
