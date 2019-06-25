from distutils.core import setup
import setuptools
setup(
    name = 'CEREBRO1',
    version = '1.11',
    license='gpl-3.0',
    description = 'recognizing facial expressions from images, videos and real-time stream',
    author = 'AmrSaber, WafaaIsmail, MohamedAhmed, SalmaSayed, MohamedAref, ManarArabi',
    author_email = 'amr.m.saber.mail@gmail.com, wafaaismail595@gmail.com, mohamedmaim97@gmail.com, Salmasayed797@gmail.com, muhammad.aref224@gmail.com, manar.araby.ma@gmail.com',
    url = 'https://github.com/AmrSaber/Cerebro',
    download_url = 'https://github.com/AmrSaber/Cerebro/archive/v1.5.tar.gz',
    keywords = ['emotions', 'expressions', 'real-time stream'],
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        'Development Status :: 5 - Production/Stable',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'License :: OSI Approved :: GNU Affero General Public License v3',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
        ],
    install_requires=[
#            'opencv-python',
            'opencv-contrib-python==3.4.4.19',
            'numpy',
            'moviepy',
            'ffmpeg-python',
            'imutils',
            'argparse',
            'keras',
            'pathlib',
            'matplotlib',
            'scikit-image',
            'dlib',
            'Theano'
            ]
)
print(setuptools.find_packages())