#modified from: https://stackoverflow.com/questions/11685716/how-to-extract-chains-from-a-pdb-file
#modify to get selected chains
import os
from Bio import PDB
import sys


class ChainSplitter:
    def __init__(self, out_dir=None):
        """ Create parsing and writing objects, specify output directory. """
        self.parser = PDB.MMCIFParser() #PDBParser()
        self.writer = PDB.PDBIO()
        if out_dir is None:
            out_dir = os.path.join(os.getcwd(), "chain_PDBs")
        self.out_dir = out_dir

    def make_pdb(self, pdb_path, chain, overwrite=False, struct=None):
        """ Create a new PDB file containing only the specified chains.

        Returns the path to the created file.

        :param pdb_path: full path to the crystal structure
        :param chain_letters: iterable of chain characters (case insensitive)
        :param overwrite: write over the output file if it exists
        """
        #chain_letters = [chain.upper() for chain in chain_letters]

        # Input/output files
        (pdb_dir, pdb_fn) = os.path.split(pdb_path)

        pdb_id = pdb_fn[:4]

        out_name = "%s_%s.ent" % (pdb_id, chain)
        out_path = os.path.join(self.out_dir, out_name)
        print("OUT PATH:",out_path)
        #plural = "s" if (len(chain_letters) > 1) else ""  # for printing

        # Skip PDB generation if the file already exists
        if (not overwrite) and (os.path.isfile(out_path)):
            print("Chain %s of '%s' already extracted to '%s'." %(chain, pdb_id, out_name))
            return out_path

        #print("Extracting chain%s %s from %s..." % (plural,
        #        ", ".join(chain_letters), pdb_fn))

        # Get structure, write new file with only given chains
        print(pdb_id, pdb_path)
        print(chain)
        if struct is None:
            struct = self.parser.get_structure(pdb_id, pdb_path)
        self.writer.set_structure(struct)
        self.writer.save(out_path, select=SelectChain(chain)) #PDB.Select.accept_chain(chain)) #SelectChains(chain))

        return out_path


class SelectChain(PDB.Select):
    """ Only accept the specified chains when saving. """
    def __init__(self, chain):
        self.chain = chain

    def accept_chain(self, chain):
        print(chain.get_id() in self.chain)
        return (chain.get_id() in self.chain)

if __name__ == "__main__":
    """ Parses PDB id's desired chains, and creates new PDB structures. """


    pdb_textfn = sys.argv[1]

    pdbList = PDB.PDBList()
    splitter = ChainSplitter(out_dir = sys.argv[2])
    pdb_dir = sys.argv[3]

    with open(pdb_textfn) as pdb_textfile:
        for line in pdb_textfile:
            pdb_id, el, chain = (line.lower()).rstrip("\n").split("\t")
            print(pdb_id, el, chain)
            #chain = line[4:]
            pdb_fn = pdbList.retrieve_pdb_file(pdb_id, pdir = pdb_dir)
            splitter.make_pdb(pdb_fn, chain)
