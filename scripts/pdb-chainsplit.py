from cgi import parse_header
from operator import index
import os
from Bio import PDB
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import sys

class ChainSelect(PDB.Select):
    def __init__(self, chain):
        self.chain = chain
    def accept_chain(self, chain):
        if chain.get_id() == self.chain:
            return 1
        else:
            return 0

def split(pdb_path, chain, start, rfam, split_dir, overwrite=False):
    cifparser = PDB.MMCIFParser(QUIET=True)
    io = PDB.MMCIFIO()
    print(pdb_path)
    
    cifdict = MMCIF2Dict(pdb_path)

    #i_chain = cifdict["_struct_asym.id"].index(chain)
    #e_id = cifdict["_struct_asym.entity_id"][i_chain]
    #fam = cifdict["_entity.pdbx_description"][int(e_id)-1]
    #print(fam)

    (pdb_dir, pdb_fn) = os.path.split(pdb_path)
    pdb_id = pdb_fn[:4]

    out_name = f"{pdb_id}_{chain}_{rfam}.cif"
    out_path = os.path.join(split_dir, out_name)
    print("OUT PATH:",out_path)

    if (not overwrite) and (os.path.isfile(out_path)):
        print(f"Chain {chain} of {pdb_id} already extracted to {out_name}.")
        return out_path

    struct = cifparser.get_structure(structure_id=pdb_id, filename=pdb_path)#[start-1]
    io.set_structure(struct)
    io.save(out_path, ChainSelect(chain))
    return

if __name__ == '__main__':
    text_file = sys.argv[1] #structure: pdb_id    pdb_start     chain   rfam_id
    pdb_dir = sys.argv[2]
    split_dir = sys.argv[3]

    pdbList = PDB.PDBList()
    with open(text_file, "r") as fh:
        for line in fh.readlines():
            pdb_id, start, chain, rfam = (line.lower()).rstrip("\n").split("\t")
            print(pdb_id, start, chain, rfam)
            pdb_path = pdbList.retrieve_pdb_file(pdb_id, pdir = pdb_dir, obsolete=False)
            split(pdb_path, chain.upper(), int(start), rfam, split_dir)

            