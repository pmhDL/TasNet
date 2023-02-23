import os.path as osp
import os
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler
from models.TasNet import TasLearner
from utils.misc import Averager, Timer, count_acc, emb_loss, compute_confidence_interval, ensure_path
from tensorboardX import SummaryWriter
from dataloader.dataset_loader import DatasetLoader as Dataset


class ConcatTrainer(object):
    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        meta_base_dir = osp.join(log_base_dir, 'concat')
        if not osp.exists(meta_base_dir):
            os.mkdir(meta_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type])
        save_path2 = 'shot' + str(args.shot) + '_way' + str(args.way) + '_query' + str(args.train_query) + \
                     '_maxepoch' + str(args.max_epoch) + '_' + args.meta_label
        args.save_path = meta_base_dir + '/' + save_path1 + '_' + save_path2
        ensure_path(args.save_path)

        # Set args to be shareable in the class
        self.args = args
        # Load train set
        self.trainset = Dataset('train', self.args)
        self.train_sampler = CategoriesSampler(self.trainset.label, self.args.num_batch, self.args.way,
                                               self.args.shot + self.args.train_query)
        self.train_loader = DataLoader(dataset=self.trainset, batch_sampler=self.train_sampler, num_workers=8,
                                       pin_memory=True)

        # Load val set
        self.valset = Dataset('val', self.args)
        self.val_sampler = CategoriesSampler(self.valset.label, 100, self.args.way,
                                             self.args.shot + self.args.val_query)
        self.val_loader = DataLoader(dataset=self.valset, batch_sampler=self.val_sampler, num_workers=8,
                                     pin_memory=True)

        # Build model
        self.model = TasLearner(self.args, mode='concat')

        # Set optimizer
        self.optimizer = torch.optim.Adam([
            # {'params': self.model.word_learner.parameters(), 'lr': self.args.meta_lr2},
            {'params': self.model.feat_learner.parameters(), 'lr': self.args.vlr}
        ], weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size,
                                                            gamma=self.args.gamma)

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()
            # if self.args.model_type == "wrn28":
            #     print("Using Parallel")
            #     self.model.encoder = torch.nn.DataParallel(self.model.encoder).cuda()


    def save_model(self, name):
        """The function to save checkpoints."""
        torch.save(dict(params=self.model.state_dict()), osp.join(self.args.save_path, name + '.pth'))

    def train(self):
        # Set the train log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0
        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0
        # Set tensorboardX
        writer = SummaryWriter(comment=self.args.save_path)

        # Generate the labels for train set of the episodes
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        if torch.cuda.is_available():
            label_shot = label_shot.type(torch.cuda.LongTensor)
        else:
            label_shot = label_shot.type(torch.LongTensor)

        # Start train
        for epoch in range(1, self.args.max_epoch + 1):
            # Update learning rate
            self.lr_scheduler.step()
            # Set the model to train mode
            self.model.train()
            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc_averager = Averager()

            # Generate the labels for test set
            label = torch.arange(self.args.way).repeat(self.args.train_query)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)

            # Using tqdm to read samples from train loader
            tqdm_gen = tqdm.tqdm(self.train_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                # Update global count number
                global_count = global_count + 1
                if torch.cuda.is_available():
                    data, _, emb = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                    emb = batch[2]
                p = self.args.shot * self.args.way
                data_shot, data_query = data[:p], data[p:]
                emb_s, emb_q = emb[:p], emb[p:]

                data_shot = F.normalize(data_shot, dim=1)
                data_query = F.normalize(data_query, dim=1)

                lg_a = self.model((data_shot, emb_s, label_shot, data_query))
                acc = count_acc(lg_a, label)
                loss = F.cross_entropy(lg_a, label)

                # Write the tensorboardX records
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)
                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f} '.format(epoch, loss.item(), acc))
                # Add loss and accuracy for the averagers
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)

                # Loss backwards and optimizer updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Update the averagers
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()
            print('    Train Average, Loss={:.4f} Acc={:.4f} '.format(train_loss_averager, train_acc_averager))
            # Start validation for this epoch, set model to eval mode
            self.model.eval()
            # Set averager classes to record validation losses and accuracies
            val_loss_averager = Averager()
            val_acc_averager = Averager()

            label = torch.arange(self.args.way).repeat(self.args.val_query)
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)

            if epoch % 10 == 0:
                print('Best Epoch {}, Best Val Acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
            # Run validation
            tqdm_gen_val = tqdm.tqdm(self.val_loader)
            for i, batch in enumerate(tqdm_gen_val, 1):
                if torch.cuda.is_available():
                    data, _, emb = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                    emb = batch[2]
                p = self.args.shot * self.args.way
                data_shot, data_query = data[:p], data[p:]
                emb_s, emb_q = emb[:p], emb[p:]
                # data_shot = F.normalize(data_shot, dim=1)
                # data_query = F.normalize(data_query, dim=1)

                lg_a = self.model((data_shot, emb_s, label_shot, data_query))
                acc = count_acc(lg_a, label)
                loss = F.cross_entropy(lg_a, label)
                tqdm_gen_val.set_description('Epoch {}, Loss={:.4f}  Acc={:.4f} '.format(epoch, loss.item(), acc))
                val_loss_averager.add(loss.item())
                val_acc_averager.add(acc)

            # Update validation averagers
            val_loss_averager = val_loss_averager.item()
            val_acc_averager = val_acc_averager.item()
            # Write the tensorboardX records
            writer.add_scalar('data/val_loss', float(val_loss_averager), epoch)
            writer.add_scalar('data/val_acc', float(val_acc_averager), epoch)
            # Print loss and accuracy for this epoch
            print('     Val Average, Loss={:.4f} Acc={:.4f} '.format(val_loss_averager, val_acc_averager))
            # Update best saved model

            if val_acc_averager > trlog['max_acc']:
                trlog['max_acc'] = val_acc_averager
                trlog['max_acc_epoch'] = epoch
                self.save_model('max_acc')
            # if epoch==self.args.max_epoch:
            #     trlog['max_acc'] = val_acc_averager
            #     trlog['max_acc_epoch'] = epoch
            #     self.save_model('max_acc')

            # Save model every 20 epochs
            if epoch % 20 == 0:
                self.save_model('epoch' + str(epoch))
            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)
            trlog['val_loss'].append(val_loss_averager)
            trlog['val_acc'].append(val_acc_averager)
            # Save log
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))
            if epoch % 10 == 0:
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(),
                                                                    timer.measure(epoch / self.args.max_epoch)))
        writer.close()


    def eval(self):
        """The function for the meta-eval phase."""
        # Load the logs
        trlog = torch.load(osp.join(self.args.save_path, 'trlog'))
        # Load test set
        test_set = Dataset('test', self.args)
        sampler = CategoriesSampler(test_set.label, 600, self.args.way, self.args.shot + self.args.val_query)
        loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
        # Set test accuracy recorder
        test_acc_record = np.zeros((600,))
        self.model.eval()
        # Set accuracy averager
        ave_acc = Averager()
        # Generate labels
        label = torch.arange(self.args.way).repeat(self.args.val_query)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        if torch.cuda.is_available():
            label_shot = label_shot.type(torch.cuda.LongTensor)
        else:
            label_shot = label_shot.type(torch.LongTensor)

        # Start test
        for i, batch in enumerate(loader, 1):
            if torch.cuda.is_available():
                data, _, emb = [_.cuda() for _ in batch]
            else:
                data = batch[0]
                emb = batch[2]
            p = self.args.shot * self.args.way
            data_shot, data_query = data[:p], data[p:]
            emb_s, emb_q = emb[:p], emb[p:]
            # data_shot = F.normalize(data_shot, dim=1)
            # data_query = F.normalize(data_query, dim=1)
            
            lg_a = self.model((data_shot, emb_s, label_shot, data_query))
            acc = count_acc(lg_a, label)
            ave_acc.add(acc)
            test_acc_record[i - 1] = acc
            if i % 100 == 0:
                print('batch {}: {:.2f}'.format(i, ave_acc.item() * 100))
        # Calculate the confidence interval, update the logs
        m, pm = compute_confidence_interval(test_acc_record)
        print('Val Best Epoch {}, Acc {:.4f}, Test Acc {:.4f} '.format(trlog['max_acc_epoch'], trlog['max_acc'], ave_acc.item()))
        print('Test Acc {:.4f} + {:.4f}'.format(m, pm))
