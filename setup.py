from distutils.core import setup
setup(
    name = 'Cerebro',
    packages = ['Cerebro'],
    version = '0.1',
    license='MIT',
    description = 'recognizing facial expressions from images, videos and real-time stream',
    long_description = read('README.md'),
    author = 'AmrSaber, WafaaIsmail, MohamedAhmed, SalmaSayed, MohamedAref, ManarArabi',
    author_email = 'amr.m.saber.mail@gmail.com, wafaaismail595@gmail.com, mohamedmaim97@gmail.com, Salmasayed797@gmail.com, muhammad.aref224@gmail.com, manar.araby.ma@gmail.com',
    url = 'https://github.com/AmrSaber/Cerebro',
    download_url = '',    # link to the last release
    keywords = ['emotions', 'expressions', 'real-time stream'],
    install_requires=[
            'opencv-python',
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
            ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        #'Topic :: Software Development :: Build Tools',
        'Programming Language :: Python :: 3.6',
        ],
)