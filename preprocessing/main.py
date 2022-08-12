from tqdm import tqdm, tnrange
import os

from functions import compute_inp_surface 

from linecache import getline
dataset_ids = os.listdir('/home/dell4/king/202112_graphDrug/data_v2/PDBbind_v2020_other_PL/v2020-other-PL_minimized/')
#line = getline('/home/dell4/king/202112_graphDrug/data_v2/PDBbind_v2020_plain_text_index/index/INDEX_refined_set.2020',10)
entity=dataset_ids[0]
#print(line)

for i in tnrange(0, len(dataset_ids)):
    entity=dataset_ids[i]
    try:
        if os.path.exists('/home/dell4/king/202112_graphDrug/data_v2/PDBbind_v2020_other_PL/v2020-other-PL_minimized/'+entity+'/'+entity+'_protein.ply'):
            print('.ply file already present:',entity)
        else:
            target_filename= '/home/dell4/king/202112_graphDrug/data_v2/PDBbind_v2020_other_PL/v2020-other-PL_minimized/'+entity+'/'+entity+'_protein.pdb'
            ligand_filename= '/home/dell4/king/202112_graphDrug/data_v2/PDBbind_v2020_other_PL/v2020-other-PL_minimized/'+entity+'/'+entity+'_ligand.mol2'
            compute_inp_surface(target_filename, ligand_filename, dist_threshold=10)
    except:
        print('Problem with file:',entity)

#print(ligand_filename, entity)
