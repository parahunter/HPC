#!/usr/bin/python

import re
import math
from subprocess import Popen, PIPE, call
import datetime

memory_approx =   [(x) for x in [4, 8, 16, 32, 64, 128, 256 , 512, 1024, 2048]] #, 4096, 8192, 16384]] 
implementation = ["nat","nmk","nkm","knm","kmn","mnk","mkn"] # "blk"] (when ready)
times = 1
 
def main():

	call("make clean", shell=True)
	call("make DRY=-xdryrun > makeinfo.txt", shell=True)
	call("make", shell=True)
	
	ns = [(int(math.ceil(math.sqrt(x*1024/(3*8))))) for x in memory_approx]
	print ns
	now = datetime.datetime.now()
	filename = "results_%s.dat" % (now.strftime("%Y-%m-%d_%H-%M"))
	print "Tests running..."
	for i in range(len(memory_approx)):
		fi = open(filename,"a+")
		fi.write(str(memory_approx[i]))
		fi.close()
		for test in implementation:
			print "Testing matmult_%s(), %d time(s), %d kB memory." % (test,times, memory_approx[i])
			process = Popen(("./matmult_f.studio %s %d %d %d" % (test,ns[i],ns[i],ns[i]+1)).split() ,stdout= PIPE,bufsize=-1) 
			(stdout,stderr) = process.communicate()
			process.wait()
			st = re.sub(' +',' ', stdout).split(" ")[1:]
			print "Result: " 
			if stdout == "":
				fi = open(filename,"a+")
				fi.write("\t")
				fi.close()
				break
			print stdout
			print "\tMemory size: %s" % st[0]
			print "\tMFLOPS     : %s" % st[1]
			print "\t", "Correct" if (st[2] == "0") else "Incorrect (error %s)" % st[2]
			
			#proto = ("%s\t" * 8) + "\n"
			#print proto % ("Impl.","m","n","k","Exp.(kB)", "Mem.(kB)", "MFLOPs", "Correct")
			#res = proto % (test,ns[i],ns[i],ns[i],memory_approx[i],st[0],st[1],st[2])
			#print res
			res = "\t" + st[1]
			fi = open(filename,"a+")
			fi.write(res)
			fi.close()
		fi = open(filename,"a+")
		fi.write("\n")
		fi.close()
	call("rm plotData.dat", shell=True)
	call(("cp %s plotData.dat" % filename), shell=True)
	call("gnuplot \"./gnuplot_config.gp\"", shell=True)
	call("lscpu > cpuinfo.txt", shell=True)
#	call("rm plotData.dat", shell=True)
if __name__ == "__main__":
	main()
