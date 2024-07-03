from setuptools import setup
import sys
sys.setrecursionlimit(sys.getrecursionlimit() * 5)
APP = ['app.py']
DATA_FILES = ['alexnet_tirads.pkl',
              'fcn_resnet50.pkl',
              'inceptionv3_simple.pkl',
              'model_inception_binary_bun.h5',
              'model_alexnet_tirads_bun.h5',
              'model_FCN_ResNet50_bun3.h5',
              'model_us_alexnet.h5',
              'right-arrow.png',
              'medical-symbol.png'
              ]
OPTIONS = {
    'iconfile': 'medical-symbol.ico',

}

setup(
    name="MATN",
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app']
)
