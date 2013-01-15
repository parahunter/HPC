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
	
	print "Tests running..."
	for N in Ns: #x axis
		thresold, iterations, timeWall, timeCPU = ({},{},{},{})
		for program in programNames:
			print "Testing N = %d, max_iterations = %d" % (N, m_fixed)
			
			origWD = os.getcwd() # remember our original working directory

			os.chdir("../../gaussSequential/")
			
			#print call(((programNames[program] + " %d %s %f") % (N,params["fixed_iterations"],m_fixed)).split())
			print ((programNames[program] + " %d %s %f") % (N,params["fixed_iterations"],m_fixed)).split()
			print os.getcwd()
			process = Popen(( ( programNames[program] + " %d %s %f ") % (N,params["fixed_iterations"],m_fixed)).split() ,stdout= PIPE,bufsize=4096) 
			(stdout,stderr) = process.communicate()
			process.wait()
			os.chdir(origWD) # get back to our original working directory


			print stdout
			thresold[program], iterations[program], timeWall[program], timeCPU[program] = parseData(stdout)

		write_row_two(filename, N, timeWall["gauss"], timeWall["gauss"]);

	os.chdir("..")


if __name__ == "__main__":
	main()
