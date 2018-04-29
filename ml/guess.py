import pathlib
import random

from ml.neural_net import Net, load_model, make_guess, get_class_list

# How-To:
#     Put .ndjson files from data.tar.gz into data/ folder.
#     Build using drawing_to_png file (change image output amount as desired).
#     Put pretrained model in models/ folder (dont change name).
#     Run guess.py
#     Model Download: https://mega.nz/#!YU8l2ChT!VEKIfNNfL7fAfoRmKFJhU7K__XTTJw2GLOUTBkFVOX8
#     data.tar.gz Download: https://mega.nz/#!l9Y01JjY!j0A4kvkN1M1Va-1uSvcO3e9fhtQnDgetIoxRQePbXqs

MODEL_NAME = 'trained_model_49.pth'  # The name of your model as found in ml/models/ folder

net = Net(total_classes=49)
load_model(net, path='models/{}'.format(MODEL_NAME))
image_to_test_path = 'images/test/banana/15001.png'
classes = get_class_list('.ndjson')
for i, c in enumerate(classes):
    print('{0:2} {1:12} '.format(i, c), end='')
    if i > 1 and i % 3 == 0:
        print('')
print('Select Class to attempt guess. ')
choice = int(input(''))
choice = classes[choice]
for p in pathlib.Path('images/test/{}/'.format(choice)).iterdir():
    image_to_test_path = str(p)
    if random.randint(0, 100) == 0:
        image_to_test_path = str(p)
        break

classes = ['apple', 'axe', 'banana', 'baseball',
           'book', 'bucket', 'car', 'cat',
           'church', 'circle', 'clock', 'cloud',
           'coffee cup', 'cookie', 'cow', 'diamond',
           'donut', 'envelope', 'fork',
           'hand', 'hexagon', 'hockey stick',
           'hourglass', 'house', 'ice cream',
           'jail', 'knife', 'ladder', 'leaf',
           'light bulb', 'lightning', 'line',
           'lollipop', 'microphone', 'moon',
           'mountain', 'oven', 'pizza',
           'power outlet', 'sandwich', 'shovel',
           'smiley face', 'square', 'star', 'sun',
           'teddy-bear', 'toaster', 'tree']

guess = make_guess(net, image_to_test_path, classes)
print('Guess: {}'.format(guess))
