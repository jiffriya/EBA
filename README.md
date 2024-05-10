# EBA
Ensemble Binding Affinity

We have used five input feature

1. Ligand atom (l)
2. Angle (a)
3. Protein (p)
4. Ligand SMILES (s)
5. Pocket (t)
   
We generated 13 models with different combination of input features Each folder name represents input feature number. For example folder name M12; M indicates model; 1 denotes ligand atom; 2 indicates angle input feature.

We implemented all 13 models in PyTorch 1.13.0 with CUDA version 11.6.
