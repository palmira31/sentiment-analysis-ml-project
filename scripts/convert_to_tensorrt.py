import subprocess

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    onnx_model_path = hydra.utils.to_absolute_path(cfg.convert.onnx_path)
    trt_model_path = hydra.utils.to_absolute_path(cfg.convert.trt_path)

    command = [
        "trtexec",
        f"--onnx={onnx_model_path}",
        f"--saveEngine={trt_model_path}",
        "--fp16",
    ]

    try:
        subprocess.run(command, check=True)
        print(f"TensorRT модель успешно сохранена: {trt_model_path}")
    except subprocess.CalledProcessError as e:
        print("Ошибка при конвертации модели:", e)


if __name__ == "__main__":
    main()
