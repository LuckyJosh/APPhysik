from setuptools import setup

setup(name="aputils",
      version="0.4.8",
      description="Collection of some useful utilities" +
                  " for the 'Anfaengerpraktikum Physik'",
      url="",
      author="Joshua Luckey",
      author_email="",
      packages=["aputils", "aputils.latextables",
                "aputils.latextables.preview"],
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Topic :: Utilities"],
      zip_safe=False)
