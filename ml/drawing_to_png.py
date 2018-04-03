"""
All the functions needed to convert a drawing (i.e. from Google Quick Draw dataset)
into a png image.
Drawings look like this: [[[224, 214, 169, 104, 57, 31, 12, 1, 1, 10, 23, 56, 100, 160, 182, 209,
236, 252, 255, 253, 245, 236, 226, 211, 211], [10, 5, 0, 2, 23, 49, 82, 116, 158, 180, 199, 227,
243, 244, 239, 226, 196, 155, 120, 79, 48, 30, 18, 7, 3]]]
LiterallyCanvas doesn't seem to have an option for outputting the
drawing strokes of a given canvas session. This file is planning ahead
in case we can't get the stroke data from literallycanvas
"""
import os
import struct
from struct import unpack
from PIL import Image
import jsonlines
import pathlib


def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    countrycode, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))

    return {
        'key_id': key_id,
        'countrycode': countrycode,
        'recognized': recognized,
        'timestamp': timestamp,
        'image': image
    }


def unpack_drawings(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield unpack_drawing(f)
            except struct.error:
                break


def unpack_ndjson(filename):
    reader = jsonlines.open(filename)
    for line in reader:
        yield line


def create_image(image, filename):
    img = Image.new('RGB', (256,256), "white")
    pixels = img.load()

    x = -1
    y = -1

    for stroke in image:
        for i in range(len(stroke[0])):
            if x != -1:
                for point in get_line(stroke[0][i], stroke[1][i], x, y):
                    pixels[point[0],point[1]] = (0, 0, 0)
            pixels[stroke[0][i],stroke[1][i]] = (0, 0, 0)
            x = stroke[0][i]
            y = stroke[1][i]
        x = -1
        y = -1
    img.save(filename)


def get_line(x1, y1, x2, y2):
    points = []
    issteep = abs(y2-y1) > abs(x2-x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2-y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x1, x2 + 1):
        if issteep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= deltay
        if error < 0:
            y += ystep
            error += deltax
    # Reverse the list if the coordinates were reversed
    if rev:
        points.reverse()
    return points


def build_image_folder(filetype='.bin', max=-1):
    """
    Takes all .bin or .ndjson files found in data folder and builds the images folder
    example structure:
        data/
            circle.bin
            square.bin
            triangle,bi

        After running build_image_list('.bin')

        images/
            circle/
                0.png
                1.png
                ...
            square/
                0.png
                1.png
                ...
            ...
    :param filetype: '.bin' or '.ndjson'
    :param max: limit of images to create (<= 0 - no limit)
    :return:
    """
    datafolder = pathlib.Path('./data')
    for i, path in enumerate(datafolder.iterdir()):
        if not path.is_dir():
            if path.name.endswith(filetype):  # Ensure desired filetype
                label = path.name[:path.name.rfind('.')]
                if not os.path.exists('./images/train'):
                    os.makedirs('./images/train')
                if not os.path.exists('./images/test'):
                    os.makedirs('./images/test')
                if not os.path.exists('./images/train/{}'.format(label)):
                    os.makedirs('./images/train/{}'.format(label))
                if not os.path.exists('./images/test/{}'.format(label)):
                    os.makedirs('./images/test/{}'.format(label))
                unpack_func, key = (unpack_drawings, 'image') if filetype in '.bin' else (unpack_ndjson, 'drawing')
                for idx, entry in enumerate(unpack_func(str(path))):
                    if 0 < max < idx:
                        if idx < max+(max*.25):
                            create_image(entry[key], './images/test/{}/{}.png'.format(label, idx))
                            continue
                        else:
                            break
                    if idx % 200 == 0:
                        print('{} {}'.format(label, idx))
                    create_image(entry[key], './images/train/{}/{}.png'.format(label, idx))
    return


build_image_folder('.ndjson', 1000)
print('finished building image folder')

# Example:
# for entry in unpack_drawings('circle.bin'):
#     create_image(entry['image'], file_output_name)
# Example using .ndjson files:
# for entry in unpack_ndjson('circle.ndjson'):
#     create_image(entry['image'], file_output_name)
