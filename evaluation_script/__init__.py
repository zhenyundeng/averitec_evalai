
# Q. How to install custom python pip packages?

# A. Uncomment the below code to install the custom python packages.

import os
import subprocess
import sys
from pathlib import Path

def install(package):
    # Install a pip python package

    # Args:
    #     package ([str]): Package name with version

    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def install_local_package(folder_name):
    # Install a local python package

    # Args:
    #     folder_name ([str]): name of the folder placed in evaluation_script/

    subprocess.check_output(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        os.path.join(str(Path(__file__).parent.absolute()) + folder_name),
    ]
)

install("scipy")
install("nltk")
install("numpy")

# install("scipy==1.10.0")
# install("nltk==3.7")
# install("numpy==1.23.5")

# install("shapely==1.7.1")
# install("requests==2.25.1")
#
# install_local_package("package_folder_name")



from .main import evaluate
