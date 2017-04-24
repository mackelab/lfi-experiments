install:
  python setup.py develop
	python setup.py build_ext --inplace

code-analysis:
	flake8 ops | grep -v __init__
