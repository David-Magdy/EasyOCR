import os
import torch.backends.cudnn as cudnn
import yaml
from train import train
from utils import AttrDict
import pandas as pd
import torch.backends.cudnn

# import mlflow 

# mlflow.set_tracking_uri("file:./mlruns")

#torch.backends.cudnn.enabled = False
cudnn.benchmark = True
cudnn.deterministic = False
def get_config(file_path):
    
    with open(file_path, 'r', encoding="utf-8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    #if opt.lang_char == 'None':
    characters = ''
    for data in opt['select_data'].split('-'):
        csv_path = os.path.join(opt['train_data'], 'labels.csv')
        df = pd.read_csv(csv_path, engine='python', usecols=['filename', 'words'], keep_default_na=False,encoding='utf-8')
        all_char = ''.join(df['words'])
        characters += ''.join(set(all_char))
    characters = sorted(set(characters))
    opt.character= ''.join(characters)
    print(f"Characters are{opt.character}")
    #else:
        # opt.character = opt.number + opt.symbol + opt.lang_char
        #opt.character = '«»؟،؛٠١٢٣٤٥٦٧٨٩' + '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ ' + '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ٠١٢٣٤٥٦٧٨٩«»؟،؛ءآأؤإئااًبةتثجحخدذرزسشصضطظعغفقكلمنهوىيًٌٍَُِّْٰٓٔٱٹپچڈڑژکڭگںھۀہۂۃۆۇۈۋیېےۓە'
    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)
    return opt

opt = get_config("config_files/ar_filtered_config5.yaml")
# with mlflow.start_run(run_name="easyocr_aug"):
#     mlflow.log_param("experiment_name", opt.experiment_name)
#     mlflow.log_param("batch_size", opt.batch_size)
#     mlflow.log_param("epochs", opt.num_iter)
#     mlflow.log_param("model", opt.Transformation + '_' + opt.FeatureExtraction)
#     mlflow.log_param("train_data", opt.train_data)
#     mlflow.log_param("valid_data", opt.valid_data)
#     mlflow.log_param("imgH", opt.imgH)
#     mlflow.log_param("imgW", opt.imgW)
#     mlflow.log_param("hidden_size", opt.hidden_size)
#     mlflow.log_param("optimizer", opt.optim)
#     mlflow.log_param("lr", opt.lr)

print("I am training now")
train(opt, amp=False)

