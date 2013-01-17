#!/usr/bin/python

import re
import math
from subprocess import Popen, PIPE, call
import datetime
import os
import sys
from testparams import *

def main():
	now = datetime.datetime.now()
	testname = "test_%s" % (now.strftime("%Y%m%d_%H%M"))
	filename = "data.dat"
	os.mkdir(testname)
	os.chdir(testname)
	call("lscpu > cpuinfo.txt", shell=True)
	N_fixed = 100
	print "Tests running..."
	for p in Ps: #x axis
		thresold, iterations, timeWall, timeCPU = ({},{},{},{})
		program = "./ompJacobiOptimized.exe"
		m = 150000
		print "Testing N = %d, max_iterations = %d" % (N_fixed, m)
		
		origWD = os.getcwd() # remember our original working directory

		os.chdir("../../ompJacobiOptimized/")
		
		print ((program + " %d %s %f") % (N_fixed,params["fixed_iterations"],m)).split()
		print os.getcwd()
		process = Popen(( ( program + " %d %s %f ") % (N_fixed,params["fixed_iterations"],m)).split() ,stdout= PIPE,bufsize=4096,env={"OMP_NUM_THREADS": str(p)}) 
		(stdout,stderr) = process.communicate()
		process.wait()
		os.chdir(origWD) # get back to our original working directory


		print stdout
		thresold[program], iterations[program], timeWall[program], timeCPU[program] = parseData(stdout)

		write_row_two(filename, p, timeWall[program],timeCPU[program]);

	os.chdir("..")


if __name__ == "__main__":
	main()
