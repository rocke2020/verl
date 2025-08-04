import os
import shutil
import sys
import requests
from loguru import logger
from safetensors import safe_open

"""  特意放在公共工具中，不同的场景适合不同的设置，openai的建议也只是一个参考，具体还是要根据实际情况来调整
https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api/172683

Use Case	Temperature	Top_p	Description
Code Generation	0.2	0.1	Generates code that adheres to established patterns and conventions. Output is more deterministic and focused. Useful for generating syntactically correct code.
Creative Writing	0.7	0.8	Generates creative and diverse text for storytelling. Output is more exploratory and less constrained by patterns.
Chatbot Responses	0.5	0.5	Generates conversational responses that balance coherence and diversity. Output is more natural and engaging.
Code Comment Generation	0.3	0.2	Generates code comments that are more likely to be concise and relevant. Output is more deterministic and adheres to conventions.
Data Analysis Scripting	0.2	0.1	Generates data analysis scripts that are more likely to be correct and efficient. Output is more deterministic and focused.
Exploratory Code Writing	0.6	0.7	Generates code that explores alternative solutions and creative approaches. Output is less constrained by established patterns.
"""

# The training config is temperture=0, top_p=1
default_infer_configs = {
    "temperature": 0.0,
    "top_p": 1,
}



def del_obsolete_models(
    root_dir: str, prefix: str | list[str], enable_del=0, del_prefix=0
):
    """Delete the obsolete trained models.

    Args:
        root_dir: the root folder of the trained models.
        prefiex: the prefix of the model dir names.
        enable_del: if 1, delete the model dirs with prefix where contain all checkpoint dir.
        del_prefix: if 1, delete the checkpoint dir with prefix; 0, delete the checkpoint dir except the prefix one
    """
    logger.info(f"{root_dir = }")
    for item in os.listdir(root_dir):
        if isinstance(prefix, str):
            prefix = [prefix]
        # delete the model dirs with prefix where contain all checkpoint dirs.
        if del_prefix:
            if any(item.startswith(p) for p in prefix) and os.path.isdir(os.path.join(root_dir, item)):
                if enable_del:
                    shutil.rmtree(os.path.join(root_dir, item))
                    logger.info(f"Deleted {item = }")
                else:
                    logger.info(f"To delete {item}, but not yet deleted.")
        # delete the checkpoint dir except the prefix one
        else:
            if any(item.startswith(p) for p in prefix) and os.path.isdir(os.path.join(root_dir, item)):
                logger.info(f"Keep {item = }")
                continue
            if enable_del:
                # remove dir
                if os.path.isdir(os.path.join(root_dir, item)):
                    shutil.rmtree(os.path.join(root_dir, item))
                # remove file
                else:
                    os.remove(os.path.join(root_dir, item))
                logger.info(f"Deleted {item = }")
            else:
                logger.info(f"To delete {item}, but not yet deleted.")


def check_safetensors_dtype(file_path):
    print(f"Inspecting SafeTensors file: {file_path}")

    with safe_open(file_path, framework="pt", device=0) as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            print(
                f"Tensor Name: {key}, Data Type: {tensor.dtype}, Shape: {tensor.shape}"
            )

    print("Inspection complete.")


if __name__ == "__main__":
    # check_safetensors_dtype(
    #     "/data/model/Qwen1.5-14B-Chat/model-00001-of-00008.safetensors"
    # )

    RUN_DEL_OBSOLETE_MODELS = 1
    if RUN_DEL_OBSOLETE_MODELS:
        del_obsolete_models(
            root_dir="checkpoints",
            prefix="0804",
            enable_del=1,
            del_prefix=0,
        )
