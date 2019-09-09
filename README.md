# Action Recognition using C3D

Pytorch Implementation 3D Convolution Neural Network (C3D) for action recognition

## 1. Prerequisites
* Opencv
* Numpy 
* Pytorch

## 2. Installation

Install required packages

```bash
pip install -r requirement.txt
```
## 3. Usage
### 3.1. Data Preparation (for training)
Download and extract UCF50 or UCF101 dataset:
* [UCF50](https://www.crcv.ucf.edu/data/UCF50.php) 
* [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

Generate dataset files (dataset.txt and categories.json)

```bash
python prepare_dataset.py -d {dataset_dir} -o {output_dir} - nf 16
```
Where:
* dataset_dir: the path to the extracted dataset
* output_dir: the path to the location of generated file
* nf: number of frame in a clip (default is 16 as in the original paper)

### 3.2. Training
* Edit the config.py file, replace the values in ```dataset_filepath``` and ```category_filepath``` with the path of the generated ```dataset.txt``` and ```categories.json``` files above
* Run the following code
```bash
python train.py
```
* Resume the training process from checkpoints:
```bash
python train.py --checkpoint {checkpoint_path}
```
Where ```checkpoint_path``` is the location of the checkpoint files
### 3.3. Pretrained networks:
* [UCF50](https://drive.google.com/file/d/1txU9fQStFHd1Q-CXqc2y_FbN1DOp_HPK/view?usp=sharing)
### 3.4. Testing
* Run the following code:
```bash
python test.py --model {model_path} --category {category_path} --video {video_path}
```
* Where:
  * ```model_path```: Path pretrained model file
  * ```category_path```: Path to the corresponding category file of the pretrained mdoel
  * ```video_path```: Path to the testing video
* By default, the output file will be saved to ```output``` folder. To change it, modify ```config.py``` and replace the value of ```output_folder``` with the desired output path.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)