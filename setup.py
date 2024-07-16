

from setuptools import setup
from pathlib import Path


long_description = (Path(__file__).parent / "README.md").read_text()


setup(
    name="python-ntp",
    version="1.0.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jarek Siembida",
    author_email="jarek.siembida@gmail.com",
    license="ASF2.0",
    url="https://github.com/jsiembida/python-ntp",
    python_requires=">=3.5",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: BSD",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Networking :: Time Synchronization",
        "Topic :: Utilities",
    ],
    zip_safe=True,
    py_modules=["ntp"],
    entry_points={
        "console_scripts": [
            "ntp = ntp:main",
        ],
    },
)
