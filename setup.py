from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'vroomba'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    (os.path.join("share", package_name, "data_structs"), glob(os.path.join('vroomba', 'data_structs', '*'))),
	(os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
	(os.path.join('share', package_name, 'models'), glob(os.path.join('models', '*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aaron David',
    maintainer_email='aarond2005@icloud.com',
    description='A node that will make a kobuki robot drive autonomously',
    license='MIT licence',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception = vroomba.map_calculator:main',
            'control = vroomba.control:main',
            'path_planning = vroomba.path_planning:main'
        ],
    },
)
