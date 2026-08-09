import os
import yaml
from typing import Dict, Any


class ConfigLoader:
    
    def __init__(self, config_path: str):

        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        config = self._resolve_dynamic_paths(config)
        
        return config
    
    def _resolve_dynamic_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:

        exp_name = config.get('model_paths', {}).get('exp_name', '')
        if not exp_name:
            return config
        
        base_path = config.get('model_paths', {}).get('base_path', '')
        path_templates = {
            'base_save_path': f"{base_path}/{exp_name}",
            'finetuned_tokenizer': f"{base_path}/{exp_name}/tokenizer/best_model"
        }
        
        if 'model_paths' in config:
            for key, template in path_templates.items():
                if key in config['model_paths']:
                    # only use template when the original value is empty string
                    current_value = config['model_paths'][key]
                    if current_value == "" or current_value is None:
                        config['model_paths'][key] = template
                    else:
                        # if the original value is not empty, use template to replace the {exp_name} placeholder
                        if isinstance(current_value, str) and '{exp_name}' in current_value:
                            config['model_paths'][key] = current_value.format(exp_name=exp_name)
        
        return config
    
    def get(self, key: str, default=None):
 
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_data_config(self) -> Dict[str, Any]:
        return self.config.get('data', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        return self.config.get('training', {})
    
    def get_model_paths(self) -> Dict[str, str]:
        return self.config.get('model_paths', {})
    
    def get_experiment_config(self) -> Dict[str, Any]:
        return self.config.get('experiment', {})
    
    def get_device_config(self) -> Dict[str, Any]:
        return self.config.get('device', {})
    
    def get_distributed_config(self) -> Dict[str, Any]:
        return self.config.get('distributed', {})
    
    def update_config(self, updates: Dict[str, Any]):

        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = update_nested_dict(self.config, updates)
    
    def save_config(self, save_path: str = None):

        if save_path is None:
            save_path = self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    def print_config(self):
        print("=" * 50)
        print("Current configuration:")
        print("=" * 50)
        yaml.dump(self.config, default_flow_style=False, allow_unicode=True, indent=2)
        print("=" * 50)


class CustomFinetuneConfig:
    
    def __init__(self, config_path: str = None):

        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        
        self.loader = ConfigLoader(config_path)
        self._load_all_configs()
    
    def _load_all_configs(self):

        data_config = self.loader.get_data_config()
        self.data_path = data_config.get('data_path')
        self.lookback_window = data_config.get('lookback_window', 512)
        self.predict_window = data_config.get('predict_window', 48)
        self.max_context = data_config.get('max_context', 512)
        self.clip = data_config.get('clip', 5.0)
        self.train_ratio = data_config.get('train_ratio', 0.9)
        self.val_ratio = data_config.get('val_ratio', 0.1)
        self.test_ratio = data_config.get('test_ratio', 0.0)
        
        # training configuration
        training_config = self.loader.get_training_config()
        # support training epochs of tokenizer and basemodel separately
        self.tokenizer_epochs = training_config.get('tokenizer_epochs', 30)
        self.basemodel_epochs = training_config.get('basemodel_epochs', 30)

        if 'epochs' in training_config and 'tokenizer_epochs' not in training_config:
            self.tokenizer_epochs = training_config.get('epochs', 30)
        if 'epochs' in training_config and 'basemodel_epochs' not in training_config:
            self.basemodel_epochs = training_config.get('epochs', 30)
        
        self.batch_size = training_config.get('batch_size', 160)
        self.log_interval = training_config.get('log_interval', 50)
        self.num_workers = training_config.get('num_workers', 6)
        self.seed = training_config.get('seed', 100)
        self.tokenizer_learning_rate = training_config.get('tokenizer_learning_rate', 2e-4)
        self.predictor_learning_rate = training_config.get('predictor_learning_rate', 4e-5)
        self.adam_beta1 = training_config.get('adam_beta1', 0.9)
        self.adam_beta2 = training_config.get('adam_beta2', 0.95)
        self.adam_weight_decay = training_config.get('adam_weight_decay', 0.1)
        self.accumulation_steps = training_config.get('accumulation_steps', 1)
        
        model_paths = self.loader.get_model_paths()
        self.exp_name = model_paths.get('exp_name', 'default_experiment')
        self.pretrained_tokenizer_path = model_paths.get('pretrained_tokenizer')
        self.pretrained_predictor_path = model_paths.get('pretrained_predictor')
        self.base_save_path = model_paths.get('base_save_path')
        self.tokenizer_save_name = model_paths.get('tokenizer_save_name', 'tokenizer')
        self.basemodel_save_name = model_paths.get('basemodel_save_name', 'basemodel')
        self.finetuned_tokenizer_path = model_paths.get('finetuned_tokenizer')
        
        experiment_config = self.loader.get_experiment_config()
        self.experiment_name = experiment_config.get('name', 'kronos_custom_finetune')
        self.experiment_description = experiment_config.get('description', '')
        self.use_comet = experiment_config.get('use_comet', False)
        self.train_tokenizer = experiment_config.get('train_tokenizer', True)
        self.train_basemodel = experiment_config.get('train_basemodel', True)
        self.skip_existing = experiment_config.get('skip_existing', False)

        unified_pretrained = experiment_config.get('pre_trained', None)
        self.pre_trained_tokenizer = experiment_config.get('pre_trained_tokenizer', unified_pretrained if unified_pretrained is not None else True)
        self.pre_trained_predictor = experiment_config.get('pre_trained_predictor', unified_pretrained if unified_pretrained is not None else True)
        
        device_config = self.loader.get_device_config()
        self.use_cuda = device_config.get('use_cuda', True)
        self.device_id = device_config.get('device_id', 0)
        
        distributed_config = self.loader.get_distributed_config()
        self.use_ddp = distributed_config.get('use_ddp', False)
        self.ddp_backend = distributed_config.get('backend', 'nccl')
        
        self._compute_full_paths()
    
    def _compute_full_paths(self):

        self.tokenizer_save_path = os.path.join(self.base_save_path, self.tokenizer_save_name)
        self.tokenizer_best_model_path = os.path.join(self.tokenizer_save_path, 'best_model')
        
        self.basemodel_save_path = os.path.join(self.base_save_path, self.basemodel_save_name)
        self.basemodel_best_model_path = os.path.join(self.basemodel_save_path, 'best_model')
    
    def get_tokenizer_config(self):

        return {
            'data_path': self.data_path,
            'lookback_window': self.lookback_window,
            'predict_window': self.predict_window,
            'max_context': self.max_context,
            'clip': self.clip,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'epochs': self.tokenizer_epochs,
            'batch_size': self.batch_size,
            'log_interval': self.log_interval,
            'num_workers': self.num_workers,
            'seed': self.seed,
            'learning_rate': self.tokenizer_learning_rate,
            'adam_beta1': self.adam_beta1,
            'adam_beta2': self.adam_beta2,
            'adam_weight_decay': self.adam_weight_decay,
            'accumulation_steps': self.accumulation_steps,
            'pretrained_model_path': self.pretrained_tokenizer_path,
            'save_path': self.tokenizer_save_path,
            'use_comet': self.use_comet
        }
    
    def get_basemodel_config(self):

        return {
            'data_path': self.data_path,
            'lookback_window': self.lookback_window,
            'predict_window': self.predict_window,
            'max_context': self.max_context,
            'clip': self.clip,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'epochs': self.basemodel_epochs,
            'batch_size': self.batch_size,
            'log_interval': self.log_interval,
            'num_workers': self.num_workers,
            'seed': self.seed,
            'predictor_learning_rate': self.predictor_learning_rate,
            'tokenizer_learning_rate': self.tokenizer_learning_rate,
            'adam_beta1': self.adam_beta1,
            'adam_beta2': self.adam_beta2,
            'adam_weight_decay': self.adam_weight_decay,
            'pretrained_tokenizer_path': self.finetuned_tokenizer_path,
            'pretrained_predictor_path': self.pretrained_predictor_path,
            'save_path': self.basemodel_save_path,
            'use_comet': self.use_comet
        }
    
    def print_config_summary(self):

        print("=" * 60)
        print("Kronos finetuning configuration summary")
        print("=" * 60)
        print(f"Experiment name: {self.exp_name}")
        print(f"Data path: {self.data_path}")
        print(f"Lookback window: {self.lookback_window}")
        print(f"Predict window: {self.predict_window}")
        print(f"Tokenizer training epochs: {self.tokenizer_epochs}")
        print(f"Basemodel training epochs: {self.basemodel_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Tokenizer learning rate: {self.tokenizer_learning_rate}")
        print(f"Predictor learning rate: {self.predictor_learning_rate}")
        print(f"Train tokenizer: {self.train_tokenizer}")
        print(f"Train basemodel: {self.train_basemodel}")
        print(f"Skip existing: {self.skip_existing}")
        print(f"Use pre-trained tokenizer: {self.pre_trained_tokenizer}")
        print(f"Use pre-trained predictor: {self.pre_trained_predictor}")
        print(f"Base save path: {self.base_save_path}")
        print(f"Tokenizer save path: {self.tokenizer_save_path}")
        print(f"Basemodel save path: {self.basemodel_save_path}")
        print("=" * 60)


class ClassifierConfig:
    """方向分类训练配置。

    读取 yaml 中的 ``classifier`` 段，支持 ``from_scratch``（方案 A）与
    ``pretrained_probe``（方案 B）两种 backbone 切换。probe 模式下若未提供
    预训练路径，主干将随机初始化（仅用于冒烟测试）。

    :param config_path: yaml 配置文件路径。
    """

    def __init__(self, config_path: str | None = None):
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "config_classifier.yaml")
        self.loader = ConfigLoader(config_path)
        self._load_configs()

    def _load_configs(self) -> None:
        cfg = self.loader.config.get("classifier", {})

        # --- 数据段 ---
        data_cfg = self.loader.config.get("data", {})
        self.data_path = data_cfg.get("data_path")
        self.lookback_window = int(data_cfg.get("lookback_window", 90))
        self.predict_window = int(data_cfg.get("predict_window", 10))
        self.clip = float(data_cfg.get("clip", 5.0))
        self.train_ratio = float(data_cfg.get("train_ratio", 0.7))
        self.val_ratio = float(data_cfg.get("val_ratio", 0.15))
        self.test_ratio = float(data_cfg.get("test_ratio", 0.15))

        # --- 主干与架构 ---
        self.backbone = cfg.get("backbone", "from_scratch")
        if self.backbone not in ("from_scratch", "pretrained_probe"):
            raise ValueError(f"未知 backbone: {self.backbone}，需为 from_scratch 或 pretrained_probe")

        arch = cfg.get("architecture", {})
        self.d_in = int(arch.get("d_in", 6))
        self.d_model = int(arch.get("d_model", 128))
        self.n_heads = int(arch.get("n_heads", 4))
        self.ff_dim = int(arch.get("ff_dim", 256))
        self.n_layers = int(arch.get("n_layers", 4))
        self.n_classes = int(arch.get("n_classes", 3))
        self.learn_te = bool(arch.get("learn_te", True))
        self.dropout = float(arch.get("dropout", 0.1))

        # probe 模式下 kronos 主干架构（与官方 Kronos-base 默认一致）
        probe_arch = cfg.get("probe_architecture", {})
        self.kronos_n_layers = int(probe_arch.get("n_layers", 12))
        self.kronos_d_model = int(probe_arch.get("d_model", 832))
        self.kronos_n_heads = int(probe_arch.get("n_heads", 16))
        self.kronos_ff_dim = int(probe_arch.get("ff_dim", 2048))
        self.kronos_s1_bits = int(probe_arch.get("s1_bits", 10))
        self.kronos_s2_bits = int(probe_arch.get("s2_bits", 10))
        # tokenizer 架构（随机初始化兜底用）
        tok_arch = cfg.get("tokenizer_architecture", {})
        self.tok_d_model = int(tok_arch.get("d_model", 256))
        self.tok_n_heads = int(tok_arch.get("n_heads", 4))
        self.tok_ff_dim = int(tok_arch.get("ff_dim", 512))
        self.tok_n_enc_layers = int(tok_arch.get("n_enc_layers", 4))
        self.tok_n_dec_layers = int(tok_arch.get("n_dec_layers", 4))

        # --- 阈值 ---
        thr = cfg.get("threshold", {})
        self.threshold_mode = thr.get("mode", "quantile")
        self.fixed_threshold = float(thr.get("fixed_threshold", 0.005))
        self.quantile_low = float(thr.get("quantile_low", 0.30))
        self.quantile_high = float(thr.get("quantile_high", 0.70))

        # --- 训练 ---
        train = cfg.get("training", {})
        self.epochs = int(train.get("epochs", 30))
        self.batch_size = int(train.get("batch_size", 32))
        self.lr = float(train.get("lr", 1e-3))
        self.adam_beta1 = float(train.get("adam_beta1", 0.9))
        self.adam_beta2 = float(train.get("adam_beta2", 0.95))
        self.adam_weight_decay = float(train.get("adam_weight_decay", 0.1))
        self.class_weighted = bool(train.get("class_weighted", True))
        self.early_stop_patience = int(train.get("early_stop_patience", 5))
        self.grad_clip = float(train.get("grad_clip", 2.0))
        self.log_interval = int(train.get("log_interval", 50))
        self.num_workers = int(train.get("num_workers", 0))
        self.seed = int(train.get("seed", 42))

        # --- 路径 ---
        paths = cfg.get("paths", {})
        self.pretrained_tokenizer = paths.get("pretrained_tokenizer", "")
        self.pretrained_predictor = paths.get("pretrained_predictor", "")
        self.save_dir = paths.get("save_dir", "./finetune_csv/finetuned/classifier")

    def print_config_summary(self) -> None:
        """打印分类训练配置摘要。"""
        print("=" * 60)
        print("Kronos 方向分类配置摘要")
        print("=" * 60)
        print(f"Backbone: {self.backbone}")
        print(f"Data path: {self.data_path}")
        print(f"Lookback/Predict: {self.lookback_window}/{self.predict_window}")
        if self.backbone == "from_scratch":
            print(f"架构: d_model={self.d_model}, n_layers={self.n_layers}, n_heads={self.n_heads}, ff_dim={self.ff_dim}")
        else:
            print(f"Probe 架构: kronos_d_model={self.kronos_d_model}, n_layers={self.kronos_n_layers}")
            print(f"预训练 tokenizer/predictor: '{self.pretrained_tokenizer}' / '{self.pretrained_predictor}'")
        print(f"阈值模式: {self.threshold_mode}, fixed={self.fixed_threshold}, quantile=({self.quantile_low},{self.quantile_high})")
        print(f"训练: epochs={self.epochs}, batch={self.batch_size}, lr={self.lr}, class_weighted={self.class_weighted}")
        print(f"Save dir: {self.save_dir}")
        print("=" * 60)
