SHELL=/bin/bash

.PHONY: load image-singularity image-docker project

all: ${EXECS}

login:
	ssh -tt rivanna "/opt/rci/bin/ijob --partition=parallel --account=bii_dsc_community --time=30:00 --ntasks-per-node=4 --nodes=2"

load:
	./load.sh

clean:
	rm -f *.log *.err script-*.slurm cylonrun-* script-*.sh script-*.lsf
	rm -r raptor-*.cfg rp.session.*

rp: 
	python rp-experiment-setup.py

cy: load
	python cylon-experiment-setup.py

q:
	bjobs --format="%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R" --me

a:
	bjobs --format="%all" --me


qq:
	watch bjobs -u arupcseuva --format=\"%.18i %.9P %.50j %.8u %.8T %.10M %.9l %.6D %R\"

i:
	cat out.log
	cat out.err
	fgrep "###"  out.log | wc -l


cancel:
	- ./cancel.sh
	- bjobs -u ${USER} 

