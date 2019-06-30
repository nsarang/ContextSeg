from collections import namedtuple


Label = namedtuple('Label', ['name', 'color'])

label_defs = [
    Label('Background', (0, 0, 0)),
    Label('Hat', (128, 0, 0)),
    Label('Hair', (255, 0, 0)),
    Label('Glove', (0, 85, 0)),
    Label('Sunglasses', (170, 0, 51)),
    Label('UpperClothes', (255, 85, 0)),
    Label('Dress', (0, 0, 85)),
    Label('Coat', (0, 119, 221)),
    Label('Socks', (85, 85, 0)),
    Label('Pants', (0, 85, 85)),
    Label('Torso-skin', (85, 51, 0)),
    Label('Scarf', (52, 86, 128)),
    Label('Skirt', (0, 128, 0)),
    Label('Face', (0, 0, 255)),
    Label('Left-arm', (51, 170, 221)),
    Label('Right-arm', (0, 255, 255)),
    Label('Left-leg', (85, 255, 170)),
    Label('Right-leg', (170, 255, 85)),
    Label('Left-shoe', (255, 255, 0)),
    Label('Right-shoe', (255, 170, 0))
]
