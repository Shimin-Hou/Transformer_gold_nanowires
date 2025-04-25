# Transformer_gold_nanowires  

Please cite [Dongying Lin, Jijie Zou, Yangyu Dong, Yudi Wang, Yongfeng Wang, Stefano Sanvitoc and Shimin Hou，Transformer-based deep learning structure–conductance relationships in gold and silver nanowires, Physical Chemstry Chemical Physics, 2025, DOI: 10.1039/d4cp04605f](https://pubs.rsc.org/en/content/articlelanding/2025/cp/d4cp04605f) for using this Transformer neural network.  

## MD  
`~/MD/AA-B-CC/` means the gold nanowire with B monatomic layers in a A×A in-plane supercell, electrodes with a cross section of C×C  
  
`T150_v2_1/` means temperature 150K, velocity 2m/s, trial No.1  
  
LAMMPS input files include:  
```
Au100_layers_test.data   # initial structure  
Au-Ag-6-0__12-16-21.pb   # machine learning force field  
in.file                  # input scripts  
```
  
MD simulation results:  
```
relax.atom   # trajectory of thermal equilibrium process  
dump.atom    # trajectory of elongating process
```

Please cite [C. M. Andolina, M. Bon, D. Passerone and W. A. Saidi, Robust, Multi-Length-Scale, Machine Learning Potential for Ag–Au Bimetallic Alloys from Clusters to Bulk Materials, J. Phys. Chem. C, 2021, 125, 17438–17447](https://pubs.acs.org/doi/10.1021/acs.jpcc.1c04403) for using this Au/Ag neural-network potential.  

  
## DFT_NEGF  
Input files for calculating Au lead are in `lead/` directory. The output `bulklft.DAT`, `bulkrgt.DAT`, `lead.DM` and `lead.HST` are used in `1_1-128/temp/.` for transport calculations.  
  
`1_1-128/` means the No.1 trajectory in training dataset, with the first to 128th snapshots.  
  
Use `smtrans2.py` to transfer MD simulation results into Smeagol input files. The output is a series of directories named 1~128. In each directory, perform DFT+NEGF calculation to obtain transmission coefficient.  
  
  
Use `coord2soap.sh` to transfer atomic structures into SOAP input data. The output is `~/Transformer/soap/soap_dat/1/`.  
Use `get_trans.sh` to collect conductance in each snapshot. The output is `~/Transformer/trans/1_tran_curve.log`.  
  
## Transformer  
Prepare training dataset `struct_655.npy` and `tran.npy`:  
```
# preprare SOAP results
cd ~/Transformer/soap/soap_dat   # There should be directories named 1~67, i.e. we have 67 MD trajectories, each contains the SOAP input data of one trajectory.  
sh create_npy.sh                 # The output is ../soap_npy/${b}/${a}_soap_result.npy
cd ../soap_npy  
python create_dateset.py         # Centralize all SOAP results in struct_655.npy   # 655 means the soap parameter rcut=6.55

# prepare conductance results
cd ~/Transformer/trans   # There should be log files named 1~67_tran_curve.log
sh cattran.sh
python cnpy.py   # The output is tran.npy
```
  
`train.py` is the main script for training.  
`model.py` contains the structures of neural networks.   
