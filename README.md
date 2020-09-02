House-GAN
======

Code and instructions for our paper:
[House-GAN: Relational Generative Adversarial Networks for Graph-constrained House Layout Generation](https://arxiv.org/pdf/2003.06988).

Data
------
![alt text](https://github.com/ennauata/housegan/blob/master/refs/sample.jpg "Sample")
[LIFULL HOME’s database](https://www.nii.ac.jp/dsc/idr/lifull) offers five million real floorplans, from which we retrieved 117,587. The database does not contain bubble diagrams. We used the floorplan vectorization algorithm [1] to generate the vector-graphics format, later converted into room bounding boxes and bubble diagrams. The vectorized floorplans utilized in this paper can be found [here](https://www.dropbox.com/sh/p707nojabzf0nhi/AAB4UPwW0EgHhbQuHyq60tCKa?dl=0), this dataset does not include the original RGB images from LIFULL dataset.<br/>
<br/>

[[1]](https://jiajunwu.com/papers/im2cad_iccv.pdf) Liu, C., Wu, J., Kohli, P., Furukawa, Y.:  Raster-to-vector:  Revisiting  floorplan transformation, ICCV 2017.

Running pretrained models
------
For running a pretrained model check out the following steps:
- Download pretrained model and dataset [here](https://www.dropbox.com/sh/p707nojabzf0nhi/AAB4UPwW0EgHhbQuHyq60tCKa?dl=0).
- Place them anywhere and rename the dataset to train_data.npy.
- Set the path in variation_bbs_with_target_graph_segments_suppl.py to the path of the folder containing train_data.npy and to the pretrained model.
- Run ***python variation_bbs_with_target_graph_segments_suppl.py***.
- Check out the results in output folder.

Training models
------
For training a model from scratch check out the following steps:
- Download dataset [here](https://www.dropbox.com/sh/p707nojabzf0nhi/AAB4UPwW0EgHhbQuHyq60tCKa?dl=0).
- Place ***housegan_clean_data.npy*** anywhere and rename it to ***train_data.npy***.
- Set the path in ***main.py*** to the path of the folder containing ***train_data.npy***.
- run ***python main.py --target_set D --exp_folder exp_example***. The target_set argument corresponds to which portion of the graphs you want to hold out for cross-validation, where D mean graphs of size 10-12. 
- You may also want to customize the interval for probing the generator by setting sample_interval in main.py.
- Check out exps and checkpoint folders for intermediate outputs and checkpoints, respectively.


Acknowledgement
------
This research is partially supported by NSERC Discovery Grants, NSERC Discovery Grants Accelerator Supplements, DND/NSERC Discovery Grant Supplement, and Autodesk. We would like to thank architects and students for participating in our user study.
