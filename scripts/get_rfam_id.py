import rna_tools.RfamSearch as rnafam
from Bio import PDB
import os
import argparse

def add_args(parser):
    parser.add_argument("-i", help="Input directory")
    parser.add_argument("-o", help="Output file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Be verbose")
    return parser

def main(args):
    cif_parser = PDB.MMCIFParser()
    desc = ""
    for file in [f for f in os.listdir(args.i) if os.path.isfile(os.path.join(args.i, f)) if f.endswith(".cif")]:
        print(file)

        structure = cif_parser.get_structure("RNA", os.path.join(args.i, file))

        seq = ""
        res_list = PDB.Selection.unfold_entities(structure, "R")

        for res in res_list:
            if not PDB.is_aa(res) and res.get_resname() in ["A", "C", "G", "U"]:
                seq += str(res.get_resname())
        
        seq = rnafam.RNASequence(seq)
        rs = rnafam.RfamSearch()
        hit = rs.cmscan(seq)
        if args.verbose:
            print(hit)
        hit = hit.split("\n")

        id_line = ""
        for line in hit:
            if line[:2] == ">>":
                id_line = line[3:]
                break
        print(id_line)
        desc += f"{file}\t{id_line}\n"
    
    if args.o:
        with open(args.o, "w") as fh:
            fh.write(desc)
    else:
        print("##"*20 +"\n")
        print("No output file specified. \nResults:\n")
        print(desc)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Searches for all .cif files in a directory the rfam database with infernal for its description. Returns the first ranked hit.")
    parser = add_args(parser)
    args = parser.parse_args()
    main(args)
    