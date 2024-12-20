from setuptools import setup, find_packages
import os

# 获取当前文件的目录
here = os.path.abspath(os.path.dirname(__file__))

# 读取版本号
version_file = os.path.join(here, 'coconut', 'version.py')
with open(version_file, 'r', encoding='utf-8') as f:
    exec(f.read())  # 这会定义 __version__

# 读取长描述从 README.md
readme_path = os.path.join(here, 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = ""

setup(
    name="CoCoNut",
    version=__version__,
    packages=find_packages(),
    install_requires=[
        "pynvml>=11.0.0",
        "torch>=1.8.0",
        "transformers>=4.0.0",
    ],
    author="HBigo",
    author_email="hbigopk@gmail.com",
    description="CoCoNut is a drinking buddy for your deep learning models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
)
