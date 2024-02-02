from distutils.core import setup
from pathlib import Path

root_dir = Path(__file__).parent # thermopack root directory
readme = (root_dir / 'README.md').read_text()

setup(name='surfpack'
	,version='v0.0.0'
	,description='Density Functional Theory for surfaces and iterfaces'
	,long_description='readme'
	,long_description_content_type='text/markdown'
	,author='Vegard Gjeldvik Jervell'
	,author_email='vegard.g.jervell@ntnu.no'
	,url='https://github.com/thermotools/surfpack'
	,packages=['surfpack']
	,package_data={'.':['solids/*']}
    ,install_requires=['numpy~=1.22',
                       'scipy~=1.7',
                       'thermopack~=2.2']
	)
