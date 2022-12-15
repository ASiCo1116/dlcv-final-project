# DLCV Final Project ( Food-classification-Challenge )

# How to run your code?
  For reproducing training phase, please run

  ```bash
  cd william
  bash run_train.sh
  ```

  Then the pretrained weight with resampling would be downloaded as 'best.pt', and a new folder name 's101_SGD_semi' would be create for saving checkpoint.

  In run_train.sh, the first line will download the pretrained weight, and the following lines are setting the parameters for training.

  ```bash
  # run_bash.sh
  wget -O 'best.pt' 'https://www.dropbox.com/s/whstfav20633buu/best.pt?dl=1'
  python3 train.py <experiment_name> \
                   --train-dir <path_to_training_folder> \
                   --valid-dir <path_to_validation_folder> \
                   --seed 1116 \
                   --gpu-ids 0 \
                   --batch-size 32 \
                   --save-freq 2 \
                   --oversampling-thr 0.0000075 \
                   --num-workers 10 \
                   --num-epochs 30 \
                   --optimizer SGD \
                   --optimizer-settings '{"lr":1e-5, "weight_decay":5e-4, "momentum":0.9}' \
                   --lr-scheduler CustomScheduler \
                   --scheduler-settings '{"0": 1e-5, "30":0}'
  ```

  For reproducing the testing result, please run

  ```bash
  python3 predict.py <output_dir> --test-dir <path_to_testing_folder> --checkpoint <path_to_checkpoint> --seed <random_seed>
  ```

    
# Usage
To start working on this final project, you should clone this repository into your local machine by using the following command:

    git clone https://github.com/DLCV-Fall-2021/final-project-challenge-3-<team_name>.git
Note that you should replace `<team_name>` with your own team name.

For more details, please click [this link](https://drive.google.com/drive/folders/13PQuQv4dllmdlA7lJNiLDiZ7gOxge2oJ?usp=sharing) to view the slides of Final Project - Food image classification. **Note that video and introduction pdf files for final project can be accessed in your NTU COOL.**

### Dataset
In the starter code of this repository, we have provided a shell script for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.

    bash ./get_dataset.sh
The shell script will automatically download the dataset and store the data in a folder called `food_data`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link](https://drive.google.com/file/d/1IYWPK8h9FWyo0p4-SCAatLGy0l5omQaw/view?usp=sharing) and unzip the compressed file manually.

> ⚠️ ***IMPORTANT NOTE*** ⚠️  
> 1. Please do not disclose the dataset! Also, do not upload your get_dataset.sh to your (public) Github.
> 2. You should keep a copy of the dataset only in your local machine. **DO NOT** upload the dataset to this remote repository. If you extract the dataset manually, be sure to put them in a folder called `food_data` under the root directory of your local repository so that it will be included in the default `.gitignore` file.

> 🆕 ***NOTE***  
> For the sake of conformity, please use the `python3` command to call your `.py` files in all your shell scripts. Do not use `python` or other aliases, otherwise your commands may fail in our autograding scripts.

# Q&A
If you have any problems related to Final Project, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under Final Project FAQ section in NTU Cool Discussion
