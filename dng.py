from flask import Flask, render_template, request
import base64
from PIL import Image

app = Flask(__name__)


def convert_image(image_path):
    img = Image.open(image_path)
    img.load()
    background = Image.new("RGB", img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])
    background.save(image_path, 'PNG')
    fin = Image.open(image_path)
    out = Image.new("RGB", img.size, (255, 255, 255))
    width, height = fin.size
    for x in range(width):
        for y in range(height):
            r, g, b = fin.getpixel((x, y))
            if r == g == b:
                if r > 150:
                    r = g = b = 255
                else:
                    r = g = b = 0
            out.putpixel((x, y), (r, g, b))
    out.save(image_path)
    return image_path


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        imgstr = request.form['base64']
        imgstr = imgstr[imgstr.find(',')+1:]
        imgdata = base64.b64decode(imgstr)
        filename = 'testing.png'
        out_path = 'images/{}'.format(filename)
        with open(out_path, 'wb') as f:
            f.write(imgdata)
        path = convert_image(out_path)
        # TODO (David-Mc): make the guess

    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    # disable debug mode for production
    app.run(debug=True)
