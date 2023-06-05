#!/usr/bin/env python

import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='numpy-illustrated',
    version='0.3.1',
    author='Lev Maximov',
    author_email='lev.maximov@gmail.com',
    url='https://github.com/axil/numpy-illustrated',
    description='Helper functions from the NumPy Illustrated guide',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires=">=3.7",
    install_requires=[
        'numpy',
    ],
    packages=['npi'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',    
        'Programming Language :: Python :: 3.10',    
        'Programming Language :: Python :: 3.11',    
    ],
    license='MIT License',
    zip_safe=False,
    keywords=['find', 'argmin', 'argmax', 'sort', 'irange', 'numpy', 'first_above', 'first_nonzero'],
)
