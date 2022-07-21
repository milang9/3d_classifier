import rna_tools.RfamSearch as rnafam
from Bio import PDB
from Bio import SeqIO
import sys

parser = PDB.MMCIFParser()

file = "/home/milan/MS_Arbeit/data/3f2q.cif"#sys.argv[1]

structure = parser.get_structure("RNA", file)


seq = ""

res_list = PDB.Selection.unfold_entities(structure, "R")

for res in res_list:
    if not PDB.is_aa(res) and res.get_resname() in ["A", "C", "G", "U"]:
        seq += str(res.get_resname())
seq = rnafam.RNASequence(seq)
rs = rnafam.RfamSearch()
hit = rs.cmscan(seq)

hit = hit.split("\n")

id_line = ""
for line in hit:
    if line[:2] == ">>":
        id_line = line
        break
print(id_line[3:])

