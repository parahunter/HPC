ms = [100*x for x in range(1,7)]
thresolds = [10**(-x) for x in range(1,5)]
Ns = [2**x for x in range(0,11)]
Ps = range(1,9)

N_fixed = 512
m_fixed = 100
thresold_fixed = 0.001
#programNames = {"jacobi" : "../../jacobiSequential/jacobiSequential.exe", "gauss" : "../../gaussSequential/gaussSequential.exe"}
programNames = {"gauss" : "./gaussSequential.exe"}

params = {"fixed_iterations" : "i", "fixed_thresold" : "e"}
unuseful_lines = 0


def write_row_two(filename,x,y1,y2):
	fi = open(filename,"a+")
	fi.write("%s\t%s\t%s\n" % (x ,y1, y2))
	fi.close()

def write_row_one(filename,x,y):
	fi = open(filename,"a+")
	fi.write("%s\t%s\n" % (x ,y))
	fi.close()

def parseData(out):
	outdata = []
	for t in out.split("\n")[unuseful_lines:]:
		if len(t.split("\t")) > 1:
			outdata.append(float(t.split("\t")[1]))
	return outdata[0], outdata[1], outdata[2], outdata[3]
