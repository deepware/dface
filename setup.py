import setuptools

with open("README.md", "r") as f:
	long_description = f.read()

setuptools.setup(
	name="dface",
	version="1.1.4",
	author="deepware",
	author_email="dogan.kurt@dodobyte.com",
	description="Face detection and recognition library that focuses on speed and ease of use.",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/deepware/dface",
	packages=setuptools.find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: BSD License",
		"Operating System :: OS Independent",
	],
)
