clean:
	rm -rf build; rm -rf dist; rm -rf *egg-info*; rm -rf *.so; find -name *.cpp -delete

init:
	mkdir pytwisterx; mkdir -p pytwisterx/common

install:
	sudo python3 setup.py install

uninstall:
	sudo pip3 uninstall -y pytwisterx;

develop:
	mkdir pytwisterx;python3 setup.py develop;

pytwisterx:
	python3 test/test_pytwisterx.py

setup:
	make clean; make uninstall; make install;

x:
	make clean; pip3 uninstall -y pytwisterx; python3 setup.py install

inplace:
	make clean; make init; python3 setup.py build_ext --inplace;
