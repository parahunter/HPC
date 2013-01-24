#!/usr/bin/python

import re
import math
from subprocess import Popen, PIPE, call
import datetime
import os
import sys

kernelSizes = [2,4,8,16,32]
matSizes = [64, 128, 512, 1024, 4096]
iterations = 10

def writeItem(filename, item):
	fi = open(filename,"a+")
	fi.write(item + "\t");
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
	filename = "PoissonTest2.dat"
	fi = open(filename,"w+")
	fi.write("");
	fi.close()
		
	print "Tests running..."		
	for kSize in kernelSizes: #x axis

		writeItem(filename, str(kSize));
		for mSize in matSizes:
			program = "./Jacobi"
				
			origWD = os.getcwd() # remember our original working directory

			os.chdir("../Poisson/")
			
			print ((program + " %d %d %d") % (mSize, iterations, kSize)).split()
			print os.getcwd()
			process = Popen(((program + " %d %d %d") % (mSize, iterations, kSize)).split() ,stdout= PIPE,bufsize=4096) 
			(stdout,stderr) = process.communicate()
			process.wait()
		
			print stdout
		
			os.chdir("../tests/")

			result = parseData(stdout)
			print result;
			writeItem(filename, result[10])
			writeItem(filename, result[12])		

		fi = open(filename,"a+")
		fi.write("\n");
		fi.close()	

	call("gnuplot \"gnuplot_PoissonTest2.gp\"", shell=True)


if __name__ == "__main__":
	main()
