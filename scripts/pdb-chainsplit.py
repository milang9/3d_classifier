import os
from Bio import PDB
import sys

def split(pdb_path, chain, start, split_dir, overwrite=False):
    cifparser = PDB.MMCIFParser(QUIET=True)
    io = PDB.PDBIO()
    (pdb_dir, pdb_fn) = os.path.split(pdb_path)
    pdb_id = pdb_fn[:4]

    out_name = f"{pdb_id}_{chain}.cif"
    out_path = os.path.join(split_dir, out_name)
    print("OUT PATH:",out_path)

    if (not overwrite) and (os.path.isfile(out_path)):
        print(f"Chain {chain} of {pdb_id} already extracted to {out_name}.")
        return out_path

    struct = cifparser.get_structure(structure_id=pdb_id, filename=pdb_path)[start-1]



    #for c in struct.get_chains():
    #    print(c, c.has_id(chain), chain)
    #    if c.has_id(chain):
    #        print("here it is")
    print(struct[chain])
    io.set_structure(struct)
    io.save(out_path, PDB.Select.accept_chain(struct[chain]))
    return



if __name__ == '__main__':
    text_file = sys.argv[1] #structure: pdb_id    pdb start     chain
    pdb_dir = sys.argv[2]
    split_dir = sys.argv[3]

    pdbList = PDB.PDBList()

    with open(text_file, "r") as fh:
        for line in fh.readlines():
            pdb_id, start, chain = (line.lower()).rstrip("\n").split("\t")
            print(pdb_id, start, chain)

            pdb_path = pdbList.retrieve_pdb_file(pdb_id, pdir = pdb_dir)
            split(pdb_path, chain.upper(), int(start), split_dir)
            sys.exit()
