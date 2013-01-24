#!/usr/bin/python

import re
import math
from subprocess import Popen, PIPE, call
import datetime
import os
import sys

matSizes = [64, 128, 256]#, 1024, 2048, 4096}
kernelSize = 16
iterations = 10

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
	filename = "MMTest.dat"
	fi = open(filename,"a+")
	fi.write("");
	fi.close()
		
	print "Tests running..."
	for size in matSizes: #x axis
		program = "./MatMult"
				
		origWD = os.getcwd() # remember our original working directory

		os.chdir("../MM/")
		
		print ((program + " %d %d %d %d %d t") % ( size, size, size, kernelSize, iterations)).split()
		print os.getcwd()
		process = Popen(( (program + " %d %d %d %d %d t") % ( size, size, size, kernelSize, iterations)).split() ,stdout= PIPE,bufsize=4096) 
		(stdout,stderr) = process.communicate()
		process.wait()
		
		print stdout
		
		os.chdir("../tests/")

		result = parseData(stdout)
		print result;
		writeRow(filename, result)

	call("gnuplot \"gnuplot_MMTest.gp\"", shell=True)


if __name__ == "__main__":
	main()
