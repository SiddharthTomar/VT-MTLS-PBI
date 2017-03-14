import sys
import subprocess
import os

def linebreaker (filename):
	z = open('prototext.txt','w')
	with open(filename) as f:
		for line in f:
			if not line.isspace():
				#sys.stdout.write(line)
					if '>' not in line:
						z.write(line)
	z.close()

def filecreater (z):
	filein = open('prototext.txt','r')
	counter = 1
	for line in filein:
		temp_line = line.rstrip()
		temporary_string = ("J" * z)+(temp_line)+("J" * z)
		fileout = open ("sample%s.fasta" % counter,'w')
		fileout.write(temporary_string)
		fileout.close()
		temporary_topology = next(filein)
		counter = counter + 1
	return (counter)
		
linebreaker ('membrane-beta_4state.3line.txt')
counter = filecreater (10)
z = 1
for i in range(counter):
	z = i + 1
	tempin = (r"C:\Users\siddh\Desktop\Project_Final\Blast\sample%s.fasta" % z)
	tempout = (r"C:\Users\siddh\Desktop\Project_Final\Blast\pssm%s.txt" % z)
	tempo = (r"C:\Users\siddh\Desktop\Project_Final\Blast\out%s.txt" % z)
	cmd = r'C:\Users\siddh\Desktop\Bioinformatique\Blast\blast-2.6.0+\bin\psiblast.exe -query {} -db C:\Users\siddh\Desktop\Project_Final\Blast\uniref50.fasta -num_iterations 3 -out_ascii_pssm {} -out {}'.format(tempin, tempout, tempo)
	print (os.system(cmd))

	

			