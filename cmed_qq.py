import re
import os
import time
import logging
import pandas as pd
from tqdm import tqdm
import numpy as np
from glob import glob
from sklearn.metrics import f1_score, classification_report

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint, EarlyStopping, global_step_from_engine
from ignite.contrib.handlers import ProgressBar

from model import CLS_Model

logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

import random
def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf``
    (if installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class cMedQQ:

    def __init__(self, dataset, model_name_or_path="bert-base-chinese", max_seq_len=128,
                 no_cuda=False, per_gpu_batch_size=8, embed_size=300, dense_layer_type="linear",
                 dropout=0.5,embed_type="random", vector_file="", bert_lr=1e-5, normal_lr=1e-3,
                 output_dir="results/cmed_qq", patience=3, n_saved=3, max_epochs=100):

        set_seed(42)
        self.max_epochs = max_epochs
        self.n_saved = n_saved
        self.patience = patience
        self.bert_lr = bert_lr
        self.normal_lr = normal_lr
        self.vector_file = vector_file
        self.model_name_or_path = model_name_or_path
        self.embed_type = embed_type
        self.dropout = dropout
        self.dense_layer_type = dense_layer_type
        self.embed_size = embed_size

        self.train_path = os.path.join(dataset, "train.csv")
        self.dev_path = os.path.join(dataset, "dev.csv")
        self.test_path = os.path.join(dataset, "test.csv")

        self.max_seq_len = max_seq_len
        self.label_list = [0, 1]
        self.per_gpu_batch_size =per_gpu_batch_size
        self.no_cuda = no_cuda


        self.bert_tokenizer = BertTokenizer.from_pretrained(model_name_or_path)

        device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
        self.n_gpu = max(torch.cuda.device_count() if not self.no_cuda else 1, 1)
        self.device = device

        if 'bert' not in self.embed_type:
            model_name = "{}_{}".format(self.embed_type, self.dense_layer_type)
        else:
            embed_type = os.path.split(self.model_name_or_path)[-1]
            model_name = "{}_{}".format(embed_type, self.dense_layer_type)

        self.model_name = model_name
        self.output_dir = "{}/{}".format(output_dir, model_name)

    def predict(self, unlabeled_path, start_time, train_time):
        all_input_ids, all_token_type_ids, \
        all_attention_mask, all_label_ids = self.get_X_y_ids(unlabeled_path)
        dataset = TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask)

        batch_size = self.n_gpu * self.per_gpu_batch_size
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

        model = CLS_Model(vocab_size=self.bert_tokenizer.vocab_size, embed_size=self.embed_size,
                          num_labels=len(self.label_list), dense_layer_type=self.dense_layer_type,
                          dropout=self.dropout, embed_type=self.embed_type, max_len=self.max_seq_len,
                          model_name_or_path=self.model_name_or_path, vector_file=self.vector_file)

        model.to(self.device)

        y_preds = []
        for model_state_path in glob(os.path.join(self.output_dir, '*{}*.pt*'.format(self.model_name))):
            model.load_state_dict(torch.load(model_state_path))
            y_pred = self.single_predict(model, dataloader)
            y_preds.append(y_pred)

        y_preds = torch.tensor(y_preds)
        y_pred = torch.mode(y_preds, dim=0).values
        y_pred = y_pred.numpy()


        report = classification_report(y_true=all_label_ids.numpy(), y_pred=y_pred, digits=4)

        predix = os.path.split(unlabeled_path)[-1].replace(".csv", "")
        score_file = os.path.join(self.output_dir, 'score_{}_{}.txt'.format(predix, self.model_name))

        data_df = pd.read_csv(unlabeled_path, names=["q1", "q2", "label", "topic"])
        data_df['pred'] = y_pred
        data_df.to_csv(os.path.join(self.output_dir, 'pred_{}_{}.csv'.format(predix, self.model_name)), index=False)

        with open(score_file, 'w', encoding="utf-8") as w:
            w.write(report)
            w.write("\n")
            w.write("train time cost:\t {:.2f} s".format(train_time))
            w.write("\n")
            w.write("time cost:\t {:.2f} s".format(time.time() - start_time - train_time))
            w.write("\n")
            w.write("args:\n{}".format('\n'.join(['%s:%s' % item for item in self.__dict__.items()])))

    def single_predict(self, model, dataloader):
        if self.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        model.eval()
        preds = None
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "token_type_ids": batch[1],
                    "attention_mask": batch[2],
                }
                _, sequence_tags = model(**inputs)

                sequence_tags = sequence_tags.detach().cpu().numpy()
                if preds is None:
                    preds = sequence_tags
                else:
                    preds = np.append(preds, sequence_tags, axis=0)
        return preds


    def train(self):
        train_input_ids, train_token_type_ids, \
        train_attention_mask, train_label_ids = self.get_X_y_ids(self.train_path)

        dev_input_ids, dev_token_type_ids, \
        dev_attention_mask, dev_label_ids = self.get_X_y_ids(self.dev_path)

        train_ds = TensorDataset(train_input_ids, train_token_type_ids, train_attention_mask, train_label_ids)
        dev_ds = TensorDataset(dev_input_ids, dev_token_type_ids, dev_attention_mask, dev_label_ids)

        batch_size = self.n_gpu * self.per_gpu_batch_size
        train_iter = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        dev_iter = DataLoader(dev_ds, batch_size=batch_size, shuffle=True, drop_last=True)

        model = CLS_Model(vocab_size=self.bert_tokenizer.vocab_size, embed_size=self.embed_size,
                          num_labels=len(self.label_list),dense_layer_type=self.dense_layer_type,
                          dropout=self.dropout,embed_type=self.embed_type,max_len=self.max_seq_len,
                          model_name_or_path=self.model_name_or_path,vector_file=self.vector_file)

        model.to(self.device)

        if self.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        logger.info("model.named_parameters()")
        for n, p in model.named_parameters():
            logger.info(n)

        parameters = [{
            "params": [p for n, p in model.named_parameters() if "bert" in n],
            "lr": self.bert_lr
        }, {
            "params": [p for n, p in model.named_parameters() if "bert" not in n],
            "lr": self.normal_lr
        }]

        optimizer = torch.optim.AdamW(parameters, lr=self.normal_lr)

        tb_writer = SummaryWriter()

        def train_fn(engine, batch):
            model.train()
            optimizer.zero_grad()
            batch = tuple(t.to(self.device) for t in batch)
            labels = batch[3]

            inputs = {
                "input_ids": batch[0],
                "token_type_ids": batch[1],
                "attention_mask": batch[2],
                "label_ids": labels
            }

            loss, sequence_tags = model(**inputs)

            score = f1_score(labels.detach().cpu().numpy(),
                             y_pred=sequence_tags.detach().cpu().numpy(), average="macro")

            if self.n_gpu > 1:
                loss = loss.mean()

            ## tensorboard
            global_step = global_step_from_engine(engine)(engine, engine.last_event_name)
            # tb_writer.add_scalar('learning_rate', scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar('train_loss', loss.item(), global_step)
            tb_writer.add_scalar('train_score', score, global_step)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            return loss.item(), score

        trainer = Engine(train_fn)
        RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'loss')
        RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'score')

        def dev_fn(engine, batch):
            model.eval()
            optimizer.zero_grad()
            with torch.no_grad():
                batch = tuple(t.to(self.device) for t in batch)
                labels = batch[3]

                inputs = {
                    "input_ids": batch[0],
                    "token_type_ids": batch[1],
                    "attention_mask": batch[2],
                    "label_ids": labels
                }

                loss, sequence_tags = model(**inputs)

            score = f1_score(labels.detach().cpu().numpy(),
                             y_pred=sequence_tags.detach().cpu().numpy(), average="macro")

            if self.n_gpu > 1:
                loss = loss.mean()

            ## tensorboard
            global_step = global_step_from_engine(engine)(engine, engine.last_event_name)
            # tb_writer.add_scalar('learning_rate', scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar('dev_loss', loss.item(), global_step)
            tb_writer.add_scalar('dev_score', score, global_step)

            return loss.item(), score

        dev_evaluator = Engine(dev_fn)
        RunningAverage(output_transform=lambda x: x[0]).attach(dev_evaluator, 'loss')
        RunningAverage(output_transform=lambda x: x[1]).attach(dev_evaluator, 'score')

        pbar = ProgressBar(persist=True, bar_format="")
        pbar.attach(trainer, ['loss', 'score'])
        pbar.attach(dev_evaluator, ['loss', 'score'])

        def score_fn(engine):
            loss = engine.state.metrics['loss']
            score = engine.state.metrics['score']
            '''
            if score < 0.5:
                logger.info("Too low to learn!")
                trainer.terminate()
            '''

            return score / (loss + 1e-12)

        handler = EarlyStopping(patience=self.patience, score_function=score_fn, trainer=trainer)
        dev_evaluator.add_event_handler(Events.COMPLETED, handler)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_dev_results(engine):
            dev_evaluator.run(dev_iter)
            dev_metrics = dev_evaluator.state.metrics
            avg_score = dev_metrics['score']
            avg_loss = dev_metrics['loss']
            logger.info(
                "Validation Results - Epoch: {}  Avg score: {:.2f} Avg loss: {:.2f}"
                    .format(engine.state.epoch, avg_score, avg_loss))

        def model_score(engine):
            score = engine.state.metrics['score']
            return score


        checkpointer = ModelCheckpoint(self.output_dir, "cmed_qq", n_saved=self.n_saved,
                                       create_dir=True, score_name="model_score",
                                       score_function=model_score,
                                       global_step_transform=global_step_from_engine(trainer),
                                       require_empty=False)

        dev_evaluator.add_event_handler(Events.COMPLETED, checkpointer,
                                        {self.model_name: model.module if hasattr(model, 'module') else model})

        # Clear cuda cache between training/testing
        def empty_cuda_cache(engine):
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)
        dev_evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)

        trainer.run(train_iter, max_epochs=self.max_epochs)

    def get_X_y_ids(self, labeled_path):
        data_df = pd.read_csv(labeled_path, names=["q1", "q2", "label", "topic"])

        all_input_ids = []
        all_token_type_ids = []
        all_attention_mask = []
        all_label_ids = []

        for q1, q2, label in tqdm(zip(data_df['q1'].values, data_df['q2'].values,
                                      data_df['label'].values), desc="Token to ids"):
            q1, q2 = self.clean_text(q1), self.clean_text(q2)
            encode_dict = self.bert_tokenizer.encode_plus(text=q1, text_pair=q2)
            input_ids = encode_dict['input_ids']
            attention_mask = encode_dict['attention_mask']
            token_type_ids = encode_dict['token_type_ids']
            padding_len = self.max_seq_len - len(input_ids)

            input_ids += [self.bert_tokenizer.pad_token_id] * padding_len
            token_type_ids += [0] * padding_len
            attention_mask += [self.bert_tokenizer.pad_token_id] * padding_len

            assert len(input_ids) == len(token_type_ids) == len(attention_mask) == self.max_seq_len

            all_input_ids.append(input_ids)
            all_token_type_ids.append(token_type_ids)
            all_attention_mask.append(attention_mask)
            all_label_ids.append(self.label_list.index(label))

        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
        all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
        all_label_ids = torch.tensor(all_label_ids)
        return all_input_ids, all_token_type_ids, all_attention_mask, all_label_ids

    def clean_text(self, text):
        def special2n(string):
            string = string.replace(r"\n", "")
            return re.sub("[ |\t|\r|\n|\\\|\u0004]", "_", string)

        def strQ2B(ustr):
            "全角转半角"
            rstr = ""
            for uchar in ustr:
                inside_code = ord(uchar)
                # 全角空格直接转换
                if inside_code == 12288:
                    inside_code = 32
                # 全角字符（除空格）根据关系转化
                elif (inside_code >= 65281 and inside_code <= 65374):
                    inside_code -= 65248

                rstr += chr(inside_code)
            return rstr

        return strQ2B(special2n(text)).lower()

    def explore_data(self, file_path):
        temp_df = pd.read_csv(file_path, names=['q1', 'q2', 'label', 'topic'])
        q1_lens = temp_df['q1'].str.len().describe()
        q2_lens = temp_df['q2'].str.len().describe()
        label_dis = temp_df['label'].value_counts()
        topic_dis = temp_df['topic'].value_counts()

        desc_path = file_path.replace("csv", 'desc')
        with open(desc_path, 'w', encoding='utf-8') as writer:
            writer.write("q1 lens distribution:\n{}".format(q1_lens))
            writer.write("\n\n")
            writer.write("q2 lens distribution:\n{}".format(q2_lens))
            writer.write("\n\n")
            writer.write("label distribution:\n{}".format(label_dis))
            writer.write("\n\n")
            writer.write("topic distribution:\n{}".format(topic_dis))

        logger.info("q1 lens distribution:\n{}".format(q1_lens))
        logger.info("q2 lens distribution:\n{}".format(q2_lens))
        logger.info("label distribution:\n{}".format(label_dis))
        logger.info("topic distribution:\n{}".format(topic_dis))

def clean_cache():
    torch.cuda.empty_cache()
    import gc
    gc.collect()

if __name__ == '__main__':
    import time

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    start_time = time.time()

    cmed_qq = cMedQQ(dataset="datasets/CMedQQ",
                     model_name_or_path="bert-base-chinese",
                     per_gpu_batch_size=32, dense_layer_type="linear",
                     embed_type="bert", bert_lr=5e-5, normal_lr=1e-4,
                     output_dir="results/cmed_qq", patience=5, max_epochs=200)
    # cmed_qq.train()
    train_time = time.time() - start_time
    cmed_qq.predict(cmed_qq.dev_path, start_time, train_time)
    train_time = time.time() - start_time
    cmed_qq.predict(cmed_qq.test_path, start_time, train_time)
    clean_cache()
