### Instructions to run the script
1. Run the following command to install all the dependencies
    ```bash
    pip install -r requirements.txt
    ```
2. Download and unzipe the backgrounds in Background folder from [this](https://drive.google.com/file/d/1JXZUiHMvPYsZiCNYlOzkbRzG2XHnDD21/view?usp=sharing) link. Create a folder named Background and unzip the folder there. Ensure that the images are in Background/Background  folder, not Background or Background/Background/Background
3. Run the following command to see all the parameters (variables) you can adjust to generate your dataset 
    ```bash
    python DataSetGeneration.py --help
    ```
4. After selecting the values, run the command with those parameters
    ```bash
    python DataSetGeneration.py --<parameter_name> <parameter_value>

    #Examples
    #Command for generating 100 images per shape of 120x120px
    python DataSetGeneration.py --n 100 --size 120
    
    #Configuration used in SUAS2024
    python DataSetGeneration.py --size 128 --n 2000 --l_prob 0.85 --s0 0.18 --s1 0.5 --b1 5 --noise 15 
    ```
5. Your dataset will be stored in the output folder

6. Run the following command to divide the generated dataset into `train`, `valid`, and `test` folders, each containing `images` and `labels` subfolders (default split: 75% train, 15% valid, 10% test)
    ```bash
    python divide_dataset.py
    ```
    The divided dataset will be saved in `output/Divided_Dataset`
   
*`DataSetGeneration.py` is the main file responsible for generating random sizes, position, rotations, colors, and letters for a particular shape and pasting it on a randomly selected background by utilizing the functions defined in `Shapes.py`. It also applies random blur and noise value, within the defined range*
