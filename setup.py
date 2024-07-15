

from setuptools import setup


setup(
    name="ntpy",
    version="1.0.0",
    description="Pure python, clean room implementation of NTP client",
    author="Jarek Siembida",
    author_email="jarek.siembida@gmail.com",
    license="ASF2.0",
    url="https://github.com/jsiembida/ntpy",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 5 - Stable",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: POSIX :: BSD",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
        "Topic :: System",
        "Topic :: System :: Networking :: Time Synchronization",
        "Topic :: Utilities",
    ],
    zip_safe=True,
    py_modules=["ntpy"],
    entry_points={
        "console_scripts": [
            "ntpy = ntpy:main",
        ],
    },
)
