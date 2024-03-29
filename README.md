# FlavorGraph
This repository provides a Pytorch implementation of **FlavorGraph2Vec**, trained on **FlavorGraph**, a large-scale food ingredient/chemical compound network of 8K nodes and 147K edges. As it is trained on not only food-food but also compound-compound relationships, our **FlavorGraph2Vec** is able to provide plausible food ingredient pairing recommendations based on chemical context. 

> **FlavorGraph: Building and embedding a large-scale food graph to suggest better food pairing** <br>
> *Donghyeon Park\*, Keonwoo Kim, Seoyoon Kim, Michael Spranger and Jaewoo Kang* <br>
> *Accepted and to be appear in [a publication or conference]* <br><br>
> *Our paper is available at:* <br>
> *[the link to the paper]* <br><br>
> You can try our demo version of **FlavorGraph**: <br>
> *[the link to the demo]*
> 
> For more details to find out what we do, please visit *https://dmis.korea.ac.kr/*

## Pipeline & Abstract
![FlavorGraph](/images/flavorgraph.png)
<p align="center">
  <b> FlavorGraph </b>
</p>

![FlavorGraph2Vec](/images/flavorgraph2vec.png)
<p align="center">
  <b> FlavorGraph2Vec Architecture </b>
</p>

![Embeddings](/images/embeddings.png)
<p align="center">
  <b> FlavorGraph2Vec Node Embeddings </b>
</p>

**Abstract** <br>
Food pairing is an area that has not yet been fully pioneered, despite our everyday experience and the large amount of data onfood. Such sophisticated and meticulous work so far has been determined by the intuition of the talented chefs. We introduceFlavorGraph that represents a large-scale food graph built on more than one million food recipes and information of 1,500flavor molecules. We analyze and extract sophisticated and meticulous relations of food both statistically and chemically so thatwe can give out a better hint of food pairing. Our graph embedding method based on deep learning surpasses other baselinemethods on food clustering. Food pairing suggestions from our model not only did it show better results, but it also showed thatseveral recommendations were possible depending on the situation. Our research offers a new reflection towards not only foodpairing techniques but also food science in general.

## Prerequisites & Development Environment
- Python 3.5.2
- PyTorch 1.0.0
- Maybe there are more. If you get an error, please try `pip install "package_name"`. 

- CUDA 9.0
- Tested on NVIDIA GeForce Titan X Pascal 12GB

## Dataset

- **[Generated pairing paths for training FlavorGraph2Vec](https://drive.google.com/file/d/1MgkxIjKUVj8yfEvB1Zh7-QP0L9iKJbEl/view?usp=sharing) (209MB)** <br>
To train the model with pre-generated pairing paths, download the above file containing user-specified paths and place it in `input/paths` folder <br> 

- **[node2fp_revised_1120.pickle]https://drive.google.com/file/d/1MPZvz6PV5yisiu2cNPRsRzH-d0ZT57Ot/view?usp=sharing) (11MB)** <br>
To train the model with Chemical Structure Prediction Layer, download the above file containing food&drug-like compound fingerprints and place it in `input` folder <br>

## Training & Test
Train the model with default settings. If you haven't download the above pairing paths file and placed it in the `input/paths` folder, the code will generate the pairing paths before running the model. The pairing paths are generated based on default settings.
```
python3 src/main.py --CSP_train --CSP_save
```
(You may remove the `--CSP_train --CSP_save` argument to train the model without Chemical Structure Prediction.)

If you want try another variation of metapaths before training the model, you can specify the arguments for the MetapathWalker as follows,
```
python3 src/main.py --number_of_walks 100 \ 
                    --walk_length 50 \ 
                    --idx_metapath 'M11' \
                    --which_metapath 'CHC+CHNHC+NHCHN' \
                    --num_walks 100 \
                    --len_metapath 50 \
                    --CSP_train --CSP_save
```

## Embeddings

After the model is trained, a pickle file containing node embeddings from FlavorGraph2Vec and their corresponding tSNE projections will be created in `output` folder. 

- **[Pickle file containing the 300D FlavorGraph node embeddings](https://drive.google.com/file/d/1MN2dGr-e8x09XSfj0kG4MahTRFY8GDw4/view?usp=sharing) (10MB)** <br>

## Contributors
**Donghyeon Park1, Keonwoo Kim2** <br>
1. Assistant Professor, FNAI Labatory, Sejong University, Seoul South Korea <br>
2. Ph.D. Candidate, DMIS Labatory, Korea University, Seoul, South Korea <br>

Please, report bugs and missing info to Donghyeon `parkdh (at) sejong.ac.kr`.

## Citation
```
@article{park2021flavorgraph,
  title={FlavorGraph: a large-scale food-chemical graph for generating food representations and recommending food pairings},
  author={Park, Donghyeon and Kim, Keonwoo and Kim, Seoyoon and Spranger, Michael and Kang, Jaewoo},
  journal={Scientific reports},
  volume={11},
  number={1},
  pages={1--13},
  year={2021},
  publisher={Nature Publishing Group}
}
```

## License
Apache License 2.0

## Changes Made
Added def export_embeddings_to_csv(embeddings, output_file):
This callable function will create a file csv file that contains vector data for given foods.

## Link to Preset data
https://drive.google.com/file/d/1eE4goYJxfgNl2gbbreHSYJ6daveC9Mz1/view?usp=drive_link
