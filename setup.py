from setuptools import setup, find_packages

setup(
    name='unet-zoo',
    version='0.1.0',
    author='Muhamad Irfan Fadhullah',
    author_email='irfanfadhullah@gmail.com',
    description='A collection of UNet variants for image segmentation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/irfanfadhullah/unet_zoo',
    packages=find_packages(where='.'),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.8.0',
        'torchvision>=0.9.0',
        'numpy',
        'scipy',
        'Pillow',
        'matplotlib',
        'pyyaml',
    ],
)