import argparse, os, math
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.contrib.examples.util import print_and_log
from pyro.infer import SVI, JitTrace_ELBO, JitTraceEnum_ELBO, Trace_ELBO, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam
from dataset_ss import DataSet, DataEntry
from vae_model import MLP, Exp, classifier_LSTM, encoder_LSTM, decoder_LSTM

def mkdir_p(path):
    os.makedirs(path)

class SSVAE(nn.Module):
    """
    This class encapsulates the parameters (neural networks) and models & guides needed to train a
    semi-supervised variational auto-encoder
    
    :param output_size: size of the tensor representing the class label (0 or 1 denotes 
    :param input_size: size of the tensor 
    :param z_dim: size of the tensor representing the latent random variable z
    """
    def __init__(self, data, output_size=2, input_size=784, z_dim=50, hidden_layers=(500,),
                 config_enum=None, use_cuda=False, aux_loss_multiplier=None):

        super(SSVAE, self).__init__()

        # initialize the class with all arguments provided to the constructor
        self.output_size = output_size
        self.input_size = input_size
        self.z_dim = z_dim
        self.hidden_layers = hidden_layers
        self.allow_broadcast = config_enum == 'parallel'
        self.use_cuda = use_cuda
        self.aux_loss_multiplier = aux_loss_multiplier

        # define and instantiate the neural networks representing
        # the paramters of various distributions in the model
        self.setup_networks(data)

    def setup_networks(self, data):

        z_dim = self.z_dim
        hidden_sizes = self.hidden_layers
        #emb_dim = self.emb_dim

        # define the neural networks used later in the model and the guide.
        # these networks are LSTMs
        self.encoder_y = classifier_LSTM(vocabulary=data.vocab, emb_dim = 128, hidden_size=64)


        # a split in the final layer's size is used for multiple outputs
        # and potentially applying separate activation functions on them
        # e.g. in this network the final output is of size [z_dim,z_dim]
        # to produce loc and scale, and apply different activations [None,Exp] on them
        self.encoder_z = encoder_LSTM(vocabulary=data.vocab, emb_dim = 128, enc_hidden_size = 64, z_hidden_size = z_dim, hidden_sizes = [32])

        self.decoder = decoder_LSTM(vocabulary=data.vocab, hidden_size=64, z_hidden_size=z_dim)

        self.one_hot_emb_x = nn.Embedding(data.vocab.count, data.vocab.count)
        self.one_hot_emb_x.weight.data = torch.eye(data.vocab.count)

        # using GPUs for faster training of the networks
        if self.use_cuda:
            self.cuda()

    def model(self, xs, ys=None):
        """
        The model corresponds to the following generative process:
        p(z) = normal(0,I)              # programming style (latent)
        p(y|x) = categorical(I/10.)     # which category (semi-supervised)
        p(x|y,z) = bernoulli(loc(y,z))   # a sequence
        loc is given by a neural network  `decoder`

        :param xs: a batch of sequence from programs
        :param ys: (optional) a batch of the class labels i.e. 1 for vulnerable 0 for safe
        :return: None
        """
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("ss_vae", self)

        max_seq_len = max([len(x) for x in xs])

        batch_size = xs.size(0)
        with pyro.plate("data"):

            # sample the programming style from the constant prior distribution
            prior_loc = xs.new_zeros([batch_size, self.z_dim]).float()
            prior_scale = xs.new_ones([batch_size, self.z_dim]).float()
            zs = pyro.sample("z", dist.Normal(prior_loc, prior_scale).to_event(1))

            # if the label y is supervised, sample from the
            # constant prior, otherwise, observe the value (i.e. score it against the constant prior)

            alpha_prior = xs.new_ones([batch_size, self.output_size]).float() / (1.0 * self.output_size)
            ys = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=ys)
            # finally, score the sequence (x) using the handwriting style (z) and
            # the class label y against the
            # parametrized distribution p(x|y,z) = bernoulli(decoder(y,z))
            # where `decoder` is a neural network
            loc = self.decoder.forward([zs, ys], max_seq_len)
            xs_oh = self.one_hot_emb_x(xs).data
            for t in range(1, max_seq_len+1):
                if (len(loc.size()) > 3):
                    pyro.sample("obs_x_%d" % t,
                            dist.Bernoulli(loc[:,:,t-1,:]).to_event(1),
                            obs=xs_oh[:, t - 1, :])
                else:
                    pyro.sample("obs_x_%d" % t,
                                dist.Bernoulli(loc[:, t - 1, :]).to_event(1),
                                obs=xs_oh[:, t - 1, :])

            # return the loc so we can visualize it later
            return loc

    def guide(self, xs, ys=None):
        """
        The guide corresponds to the following:
        q(y|x) = categorical(alpha(x))              # infer class
        q(z|x,y) = normal(loc(x,y),scale(x,y))       # infer programming style from a sequence and the class label
        loc, scale are given by a neural network `encoder_z`
        alpha is given by a neural network `encoder_y`

        :param xs: a batch of sequence from programs
        :param ys: (optional) a batch of the class labels i.e. 1 for vulnerable 0 for safe
        :return: None
        """
        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate("data"):

            # if the class label is not supervised, sample
            # (and score) the class with the variational distribution
            # q(y|x) = categorical(alpha(x))
            if ys is None:
                alpha = self.encoder_y.forward(xs)
                ys = pyro.sample("y", dist.OneHotCategorical(alpha))

            # sample (and score) the latent programming-style with the variational
            # distribution q(z|x,y) = normal(loc(x,y),scale(x,y))
            loc, scale = self.encoder_z.forward([xs, ys])
            pyro.sample("z", dist.Normal(loc, scale).to_event(1))

    def classifier(self, xs):
        """/
        classify a sequence

        :param xs: a batch of sequence from programs
        :return: a batch of the corresponding class labels (as one-hots)
        """
        alpha = self.encoder_y.forward(xs)

        res, ind = torch.topk(alpha, 1)

        return ind.squeeze().float()

    def model_classify(self, xs, ys=None):
        """
        this model is used to add an auxiliary (supervised) loss as described in the
        Kingma et al., "Semi-Supervised Learning with Deep Generative Models".
        """
        # register all pytorch (sub)modules with pyro
        pyro.module("ss_vae", self)

        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate("data"):
            # this here is the extra term to yield an auxiliary loss that we do gradient descent on
            if ys is not None:
                alpha = self.encoder_y.forward(xs)
                with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                    pyro.sample("y_aux", dist.OneHotCategorical(alpha), obs=ys)

    def guide_classify(self, xs, ys=None):
        """
        dummy guide function to accompany model_classify in inference
        """
        pass


def run_inference_for_epoch(data_loaders, losses, periodic_interval_batches, one_hot_emb_y):
    """
    runs the inference algorithm for an epoch
    returns the values of all losses separately on supervised and unsupervised parts
    """
    num_losses = len(losses)

    # compute number of batches for an epoch
    sup_batches = len(data_loaders.batch_indices)
    unsup_batches = len(data_loaders.unsup_batch_indices)
    batches_per_epoch = sup_batches + unsup_batches
    print("BATCH:", sup_batches, unsup_batches)

    # initialize variables to store loss values
    epoch_losses_sup = [0.] * num_losses
    epoch_losses_unsup = [0.] * num_losses

    # count the number of supervised batches seen in this epoch
    ctr_sup = 0

    for i in range(batches_per_epoch):
        # whether this batch is supervised or not
        is_supervised = (i % periodic_interval_batches == 1) and ctr_sup < sup_batches
        
        # extract the corresponding batch
        if is_supervised:
            sequences, masks, labels, seq_len = data_loaders.get_next_batch_train_data()
            labels = one_hot_emb_y(labels).data
            ctr_sup += 1
        else:
            sequences, masks, labels, seq_len = data_loaders.get_next_batch_train_unsup_data()
        # run the inference for each loss with supervised or un-supervised
        # data as arguments
        for loss_id in range(num_losses):
            if is_supervised:
                new_loss = losses[loss_id].step(sequences, labels)
                epoch_losses_sup[loss_id] += new_loss
            else:
                new_loss = losses[loss_id].step(sequences)
                epoch_losses_unsup[loss_id] += new_loss

    # return the values of all losses
    return epoch_losses_sup, epoch_losses_unsup


def get_accuracy(data_loader, classifier_fn, batch_size):
    """
    compute the accuracy over the supervised training set or the testing set
    """
    predictions, actuals = [], []
    test_batches = len(data_loader.test_batch_indices)

    # use the appropriate data loader

    for i in range(test_batches):
        sequences, masks, labels = data_loader.get_next_batch_test_data()
        # use classification function to compute all predictions for each batch
        predictions.append(classifier_fn(sequences))
        actuals.append(labels)

    # compute the number of accurate predictions
    accurate_preds = 0
    for pred, act in zip(predictions, actuals):
        for i in range(pred.size(0)):
            print(pred[i], act[i])
            v = torch.sum(pred[i] == act[i])
            accurate_preds += v.item()

    # calculate the accuracy between 0 and 1
    accuracy = (accurate_preds * 1.0) / (len(predictions) * batch_size)
    data_loader.initialize_test_batch()
    return accuracy



def main(args):
    """
    run inference for SS-VAE
    :param args: arguments for SS-VAE
    :return: None
    """
    if args.seed is not None:
        pyro.set_rng_seed(args.seed)

    parser = None
    data = DataSet()
    data_file =  './data/juliet.data'
    label_file = './data/juliet.label'

    #Unsupervised dataset
    wild_data_file = ['./data/cwe119.data', './data/cwe399.data'] #just for test, it's not a wild dataset
    wild_label_file = ['./data/cwe119.label', './data/cwe399.label']

    # Add Supurvised
    with open(data_file) as df:
        with open(label_file) as dl:
            for line, label in zip(df, dl):
                entry = DataEntry(data, line.strip(), int(label), meta_data = -1)
                data.add_data_entry(entry)

    # Add Unsurpurvised Data
    for wild_df, wild_label in zip(wild_data_file, wild_label_file):
        with open(wild_df) as df:
            with open(wild_label) as dl:
                for line, label in zip(df, dl):
                    entry = DataEntry(data, line.strip(), label = int(label))
                    data.add_data_entry(entry)

    data.init_data_set(vocab_size=1000, batch_size = args.batch_size, test_percentage=0.1)

    print(data.sup_num, data.train_data_size)

    # batch_size: number of samples to be considered in a batch
    ss_vae = SSVAE(data = data, z_dim=args.z_dim,
                   hidden_layers=args.hidden_layers,
                   use_cuda=args.cuda,
                   config_enum=args.enum_discrete,
                   aux_loss_multiplier=args.aux_loss_multiplier)

    one_hot_emb_y = nn.Embedding(2, 2).cuda()
    one_hot_emb_y.weight.data = torch.eye(2).cuda()

    # setup the optimizer
    adam_params = {"lr": args.learning_rate, "betas": (args.beta_1, 0.999)}
    optimizer = Adam(adam_params)

    # set up the loss(es) for inference. wrapping the guide in config_enumerate builds the loss as a sum
    # by enumerating each class label for the sampled discrete categorical distribution in the model
    guide = config_enumerate(ss_vae.guide, args.enum_discrete, expand=True)
    elbo = (JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO)(max_plate_nesting=1)
    loss_basic = SVI(ss_vae.model, guide, optimizer, loss=elbo)

    # build a list of all losses considered
    losses = [loss_basic]

    # aux_loss: whether to use the auxiliary loss from NIPS 14 paper (Kingma et al)
    if args.aux_loss:
        elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
        loss_aux = SVI(ss_vae.model_classify, ss_vae.guide_classify, optimizer, loss=elbo)
        losses.append(loss_aux)

    # setup the logger if a filename is provided
    logger = open(args.logfile, "w") if args.logfile else None


    # how often would a supervised batch be encountered during inference
    # e.g. if sup_num is 3000, we would have every 16th = int(50000/3000) batch supervised
    # until we have traversed through the all supervised batches
    periodic_interval_batches = int(math.ceil(data.train_data_size / (1.0 * data.sup_num)))

    print(periodic_interval_batches)

    # number of unsupervised examples
    unsup_num = data.train_data_size - data.sup_num
    best_valid_acc, corresponding_test_acc = 0.0, 0.0

    # run inference for a certain number of epochs
    for i in range(0, args.num_epochs):
        # get the losses for an epoch
        epoch_losses_sup, epoch_losses_unsup = \
            run_inference_for_epoch(data, losses, periodic_interval_batches, one_hot_emb_y)

        data.initialize_batch()
        data.initialize_unsup_batch()

        # compute average epoch losses i.e. losses per example
        avg_epoch_losses_sup = map(lambda v: v / data.sup_num, epoch_losses_sup)
        avg_epoch_losses_unsup = map(lambda v: v / unsup_num, epoch_losses_unsup)

        # store the loss and validation/testing accuracies in the logfile
        str_loss_sup = " ".join(map(str, avg_epoch_losses_sup))
        str_loss_unsup = " ".join(map(str, avg_epoch_losses_unsup))

        str_print = "{} epoch: avg losses {}".format(i, "{} {}".format(str_loss_sup, str_loss_unsup))

        # this test accuracy is only for logging, this is not used
        # to make any decisions during training
        torch.save(ss_vae, './vaeo_' +str(i) + '.pt')
        test_accuracy = get_accuracy(data, ss_vae.classifier, args.batch_size)
        str_print += " test accuracy {}".format(test_accuracy)
        corresponding_test_acc = test_accuracy

        print_and_log(logger, str_print)

    final_test_accuracy = get_accuracy(data, ss_vae.classifier, args.batch_size)
    print_and_log(logger, "best validation accuracy {} corresponding testing accuracy {} "
                          "last testing accuracy {}".format(best_valid_acc, corresponding_test_acc,
                                                            final_test_accuracy))
    if args.logfile:
        logger.close()



EXAMPLE_RUN = "example run: python ss_vae.py --seed 0 --cuda -n 2 --aux-loss -alm 46 -enum parallel " \
              "-sup 3000 -zd 50 -hl 500 -lr 0.00042 -b1 0.95 -bs 200 -log ./tmp.log"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SS-VAE\n{}".format(EXAMPLE_RUN))

    parser.add_argument('--cuda', action='store_true', default=True,
                        help="use GPU(s) to speed up training")
    parser.add_argument('--jit', action='store_true', default=False,
                        help="use PyTorch jit to speed up training")
    parser.add_argument('-n', '--num-epochs', default=10, type=int,
                        help="number of epochs to run")
    parser.add_argument('--aux-loss', action="store_true", default=True,
                        help="whether to use the auxiliary loss from NIPS 14 paper "
                             "(Kingma et al). It is not used by default ")
    parser.add_argument('-alm', '--aux-loss-multiplier', default=100,
                        help="the multiplier to use with the auxiliary loss")
    parser.add_argument('-enum', '--enum-discrete', default="parallel",
                        help="parallel, sequential or none. uses parallel enumeration by default")
    parser.add_argument('-sup', '--sup-num', default=3000,
                        help="supervised amount of the data")
    parser.add_argument('-zd', '--z-dim', default=64, type=int,
                        help="size of the tensor representing the latent variable z "
                             "variable")
    parser.add_argument('-hl', '--hidden-layers', nargs='+', default=[500], type=int,
                        help="a tuple (or list) of MLP layers to be used in the neural networks "
                             "representing the parameters of the distributions in our model")
    parser.add_argument('-lr', '--learning-rate', default=0.001,
                        help="learning rate for Adam optimizer")
    parser.add_argument('-b1', '--beta-1', default=0.9,
                        help="beta-1 parameter for Adam optimizer")
    parser.add_argument('-bs', '--batch-size', default=32, type=int,
                        help="number of images (and labels) to be considered in a batch")
    parser.add_argument('-log', '--logfile', default="./tmp.log", type=str,
                        help="filename for logging the outputs")
    parser.add_argument('--seed', default=42, type=int,
                        help="seed for controlling randomness in this example")
    args = parser.parse_args()

    main(args)
