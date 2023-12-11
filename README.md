# Augmenting-Embodied-Scenes
This repository provides a robust framework for applying consistent augmentation to Embodied AI environments using ControlNet. It streamlines the process, ensuring uniformity & consistency and leveraging the power of ControlNet to enhance the augmentation workflow.

Original          |  Edited
:-------------------------:|:-------------------------:
![orig_deneme (1)](https://github.com/yesiltepe-hidir/Augmenting-Embodied-Scenes/assets/70890453/fd885468-ea08-4c30-a40a-d4453eb2355b)   |  ![result_deneme (1)](https://github.com/yesiltepe-hidir/Augmenting-Embodied-Scenes/assets/70890453/1ce9e1d0-4291-4609-9019-238cd234a5a8)

# Run
To be able augment to scene:

```
# Create environment
conda env create -f environment.yaml
```

Open the `playground.ipynb` with `allenact` virtual environment. 

Run `Save Frames` cell to extract original scene frames.

Run `Augmentation of Scenes` cell to edit original scene frames.

We also share all the metrics we used in our experimentation in 'metrics.ipynb' file.

# Contributors
Kiymet Akdemir, Yusuf Dalva, Tuna Meral, Hidir Yesiltepe
