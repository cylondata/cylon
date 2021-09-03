clean:
	rm -rf build; rm -rf dist; rm -rf *egg-info*; rm -rf *.so;

init:

install:
	sudo python3 setup.py install

uninstall:
	sudo pip3 uninstall -y pygcylon;

develop:


setup:
	make clean; make uninstall; make install;

x:
	make clean; pip3 uninstall -y pygcylon; python3 setup.py install

inplace:
	make clean; make init; python3 setup.py build_ext --inplace;
