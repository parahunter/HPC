#!/usr/bin/python

import re
import math
from subprocess import Popen, PIPE, call
import datetime
import os
import sys

matSizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
proci = [1,2,4,6,8,10,12]
kernelSize = 16
#iterations = 10
iterat = [10, 50, 100,500,1000]

def writeRow(filename, parseData):
	fi = open(filename,"a+")
	
	for data in parseData:
		fi.write("%s \t" % (data))

	fi.write("\n");
	fi.close()

unuseful_lines = 0
def parseData(out):
	outdata = []
	for t in out.split("\n")[unuseful_lines:]:
		if len(t.split("\t")) > 1:
			line = t.split("\t")
			for val in line:
				outdata.append(val)
	return outdata;

def main():
	filename = "PoissonTest1.dat"
	fi = open(filename,"w+")
	fi.write("");
	fi.close()
		
	print "Tests running..."
	#size=1024
	iterations = 200
#	for iterations in iterat:
	for size in matSizes: #x axis
		program = "./Jacobi"
				
		origWD = os.getcwd() # remember our original working directory

		os.chdir("../Poisson/")
		
		print ((program + " %d %d %d") % (size, iterations, kernelSize)).split()
		print os.getcwd()
		process = Popen(((program + " %d %d %d") % (size, iterations, kernelSize)).split() ,stdout= PIPE,bufsize=4096,env={"OMP_NUM_THREADS": str(8)}) 
		(stdout,stderr) = process.communicate()
		process.wait()
		
		#print stdout
		
		os.chdir("../tests/")
		print stdout
		result = parseData(stdout)
		
		writeRow(filename, result)
	
	call("gnuplot \"gnuplot_PoissonTest1Times.gp\"", shell=True)


if __name__ == "__main__":
	main()
