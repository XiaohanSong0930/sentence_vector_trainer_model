# coding=utf8
import os
import random
import sys
import logging
import yaml
import torch
from torch.utils.data import Dataset
from os.path import join
from copy import deepcopy
from transformers import AutoTokenizer, AutoModel
from transformers import TrainingArguments, Trainer
import shutil

os.environ["WANDB_DISABLED"] = "true"
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
from transformers import TrainerCallback, TrainerControl, TrainerState
import json
import torch.nn.functional as F


class SaveModelCallBack(TrainerCallback):
    def __init__(self, output_dir, save_steps):
        self.customized_save_steps = save_steps
        self.customized_output_dir = output_dir

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # print(f"-R{local_rank}-S{state.global_step}-")
        if local_rank == 0 and self.customized_save_steps > 0 and state.global_step > 0 and state.global_step % self.customized_save_steps == 0:
            epoch = int(state.epoch)
            save_dir = join(self.customized_output_dir, f"epoch-{epoch}_globalStep-{state.global_step}")
            kwargs["model"].save_pretrained(save_dir, max_shard_size="900000MB")
            kwargs["tokenizer"].save_pretrained(save_dir)
            kwargs["tokenizer"].save_vocabulary(save_dir)

class MyTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        def get_vecs(ipt):
            attention_mask = ipt["attention_mask"]
            model_output = model(**ipt)
            last_hidden = model_output.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
            vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
            vectors = F.normalize(vectors, 2.0, dim=1)
            return vectors

        q_num = inputs[-1]
        inputs = inputs[:-1]
        if model_name == "e5":
            vectors = [get_vecs(ipt) for ipt in inputs]
            vectors = torch.cat(vectors, dim=0)
            # 计算loss
            vecs1, vecs2 = vectors[:q_num, :], vectors[q_num:, :]
            logits = torch.mm(vecs1, vecs2.t())
            # print(logits.shape)
            LABEL = torch.LongTensor(list(range(q_num))).to(vectors.device)
            vec_loss = F.cross_entropy(logits * 10, LABEL)
        else:
            raise NotImplementedError()
        return (vec_loss, None) if return_outputs else vec_loss


def collate_fn(batch):
    """

    :param batch:List[data_set[i]]
    :return:
    """
    # 随机获取一批难负例
    hard_negs = [random.choice(negs) for _, _, negs in batch if negs]
    batch = [[t1, t2] for t1, t2, _ in batch]
    batch = list(dict(batch).items())
    batch = [item[::-1] for item in batch]
    batch = list(dict(batch).items())
    batch = [item[::-1] for item in batch]

    pos_texts = set([j for item in batch for j in item])
    hard_negs = [i for i in hard_negs if i not in pos_texts]
    all_texts = [item[0] for item in batch] + [item[1] for item in batch] + hard_negs
    ipts = []
    for start in range(0, len(all_texts), 32):
        ipt = tokenizer.batch_encode_plus(
            all_texts[start:start + 32], padding="longest", truncation=True, max_length=max_length, return_tensors="pt")
        ipts.append(ipt)
    # 最后把q数量加上
    ipts.append(len(batch))
    return ipts


class VectorDataSet(Dataset):
    def __init__(self, data_paths: str):
        self.data = []
        self.data_paths = data_paths
        self.load_data()

    def load_data(self):
        for data_path in self.data_paths:
            with open(data_path, "r", encoding="utf8") as fr:
                single_data = [json.loads(line) for line in fr]
                if isinstance(data_tag_filter, list):
                    single_data = [item for item in single_data if item["where"] in data_tag_filter]
                self.data.extend(single_data)
        self.data = [[item["txt1"], item["txt2"], item.get("hard_negs", [])] for item in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        """
        item 为数据索引，迭代取第item条数据
        """

        return self.data[item]


MODEL_NAME_INFO = {
    "e5": [AutoModel, AutoTokenizer, collate_fn],
}

if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # 读取参数并赋值
    with open(sys.argv[1].strip(), "r", encoding="utf8") as fr:
        conf = yaml.safe_load(fr)

    # args of hf trainer
    hf_args = deepcopy(conf["train_args"])
    if hf_args.get("deepspeed") and conf["use_deepspeed"]:
        hf_args["deepspeed"]["gradient_accumulation_steps"] = hf_args["gradient_accumulation_steps"]
        hf_args["deepspeed"]["train_micro_batch_size_per_gpu"] = hf_args["per_device_train_batch_size"]
        hf_args["deepspeed"]["optimizer"]["params"]["lr"] = hf_args["learning_rate"]
    else:
        hf_args.pop("deepspeed", None)

    model_name = conf["model_name"]

    grad_checkpoint = hf_args["gradient_checkpointing"]
    task_name = conf["task_name"]
    train_paths = conf["train_paths"]
    dev_paths = conf["dev_paths"]
    max_length = conf["max_length"]
    model_dir = conf["model_dir"]
    train_method = conf["train_method"]
    data_tag_filter = conf["data_tag_filter"]

    output_dir = hf_args["output_dir"]

    # 构建训练输出目录
    if world_size == 1:
        version = 1
        save_dir = join(output_dir, f"{model_name}_{task_name}_v{version}")
        while os.path.exists(save_dir):
            version += 1
            save_dir = join(output_dir, f"{model_name}_{task_name}_v{version}")
        output_dir = save_dir
        hf_args["output_dir"] = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        # TODO 多卡训练的时候，会出现问题,所以多卡必须提前写好提前设定好
        if not os.path.exists(hf_args["output_dir"]):
            os.makedirs(hf_args["output_dir"], exist_ok=True)
    # 拷贝 config
    if local_rank == 0:
        shutil.copy(sys.argv[1].strip(), os.path.join(output_dir, "train_config.yml"))
        # 初始化log
        logging.basicConfig(level=logging.INFO, filename=join(output_dir, "train_log.txt"), filemode="a")
    coll_fn = MODEL_NAME_INFO[model_name][2]
    # train 数据集
    train_dataset = VectorDataSet(
        data_paths=train_paths,
    )
    # dev 数据集
    dev_dataset = VectorDataSet(
        data_paths=dev_paths,
    )
    # 加载模型、tokenizer
    model = MODEL_NAME_INFO[model_name][0].from_pretrained(
        model_dir,
        torch_dtype=torch.float16
    )
    tokenizer = MODEL_NAME_INFO[model_name][1].from_pretrained(model_dir, trust_remote_code=True)
    if grad_checkpoint:
        try:
            model.gradient_checkpointing_enable()
        except:
            logging.error("gradient_checkpointing failed")
    model.train()

    # 开始训练
    args = TrainingArguments(
        **hf_args,
        torch_compile=torch.__version__.startswith("2"),
        prediction_loss_only=True,
    )
    # TODO do not know why, can solve PyTorch DDP: Finding the cause of "Expected to mark a variable ready only once"
    if hf_args["gradient_checkpointing"]:
        args.ddp_find_unused_parameters = False
    # save model by call back, do not need official save function
    args.save_strategy = "no"
    trainer = MyTrainer(
        model=model,
        args=args,
        data_collator=coll_fn,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        callbacks=[SaveModelCallBack(output_dir=output_dir, save_steps=conf["save_steps"])]
    )
    trainer.train()
