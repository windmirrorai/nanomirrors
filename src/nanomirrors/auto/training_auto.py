from .registry import resolve_class

from typing import Optional, Any, Dict

import logging

class AutoTrainerConfig:

    @classmethod
    def from_model(cls, model_name: str, task_name: str = "perturbation_response_prediction", **kwargs):
        trainer_config_class = resolve_class(model_name, "trainer_config", task_name=task_name)
        return trainer_config_class(**kwargs)

class AutoTrainerForPerturbationResponsePrediction:

    @classmethod
    def from_model(cls, 
                   model_name: str, 
                   n_genes: int,
                   *,
                   trainer_config_params: Optional[Dict[str, Any]] = None,
                   model_config_params: Optional[Dict[str, Any]] = None,
                   device: Optional[str] = 'cpu',
                   logger: Optional[logging.Logger] = None,
                   save_dir: Optional[str] = None):


        trainer_config = AutoTrainerConfig.from_model(
            model_name,
            task_name="perturbation_response_prediction",
            **(trainer_config_params or {}),
        )
        model_class = resolve_class(model_name, "model", task_name="perturbation_response_prediction")
        model = model_class.from_pretrained(n_genes=n_genes, 
                                            model_name=model_name,
                                            **(model_config_params or {}))
        model.to(device)

        
        trainer_class = resolve_class(model_name, "trainer", task_name="perturbation_response_prediction")
        trainer = trainer_class(model=model,
                config=trainer_config,
                device=device,
                logger=logger,
                save_dir=save_dir)

        # print('Trainer Config: ', trainer_config.__dict__)
        # print('Model Config: ', model.config.__dict__)
        
        return trainer
        