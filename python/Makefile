clean:
	rm -rf build; rm -rf dist; rm -rf *egg-info*; rm -rf *.so; rm -rf pytwisterx; rm -rf twisterx/*/*.cpp;

install:
	python3 setup.py install

uninstall:
	pip3 uninstall pytwisterx

develop:
	mkdir pytwisterx;python3 setup.py develop;

pytwisterx:
	python3 test/test_pytwisterx.py

clear:
	make clean; make uninstall; make develop; make install;
