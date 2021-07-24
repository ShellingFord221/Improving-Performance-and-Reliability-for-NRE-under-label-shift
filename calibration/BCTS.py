import torch
from torch import optim
from torch.nn import functional as F
import torch.nn as nn



class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, label_num, device):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.model.eval()

        self.device = device
        self.cpu = torch.device('cpu')

        self.temperature = nn.Parameter(torch.ones(1) * 1.5)   # BCTS
        self.systematic_bias = nn.Parameter(torch.zeros(label_num))   # 共label_num个类
        
        self.optimal_temp = 0
        self.optimal_bias = 0

    def forward(self):
        pass

    # 用于记下最佳参数
    def set_best_parameter(self, logits, temp, bias):
        self.optimal_temp = temp.cuda()
        self.optimal_bias = bias.cuda()
        return logits / self.optimal_temp + self.optimal_bias

    # 用于调参
    def scale_temperature(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        # isotonic
        temperature = self.temperature.cuda()
        # anisotropic
        systematic_bias = self.systematic_bias.cuda()
        return logits / temperature + systematic_bias

    # 用于calibrate (best parameter)
    def calibrate_by_temperature(self, logits):
        return logits / self.optimal_temp + self.optimal_bias


    # This function probably should live outside of this class, but whatever
    def set_temperature(self, dset):
        """
        Tune the temperature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        logits_list = []
        labels_list = []
        with torch.no_grad():
            # for input, label in valid_loader:
            for idx, batch in enumerate(dset):

                # input = [p.cuda() for p in batch[:5]]
                # label = batch[2]

                b_input_ids = batch[0].cuda()
                b_input_mask = batch[1].cuda()
                b_labels = batch[2].cuda()
                b_e1_pos = batch[3].cuda()
                b_e2_pos = batch[4].cuda()


                loss, logits = self.model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels,
                                    e1_pos=b_e1_pos,
                                    e2_pos=b_e2_pos)

                logits_list.append(logits)
                labels_list.append(b_labels)

            print(len(logits_list))
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()
            # print(logits_list.size())

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece, before_temperature_acc = ece_criterion(logits, labels)
        print('Before temperature - NLL: %.3f, ECE: %.3f, ACC: %.3f' % (before_temperature_nll, before_temperature_ece.item(), before_temperature_acc))

        # ece_original = self.make_model_diagrams(logits, labels, fig_name + '_original')
        # print(ece_original)


        optimizer = optim.Adam([self.temperature, self.systematic_bias], lr=0.001)
        # calibrate_f1 = 0
        best_loss = 10000
        optimal_temp = 0
        optimal_bias = 0
        for i in range(10000):
            optimizer.zero_grad()
            # loss, _ = ece_criterion(self.scale_temperature(logits), labels)
            scaled_logits = self.scale_temperature(logits)
            loss = nll_criterion(scaled_logits, labels)
            _, preds = torch.max(scaled_logits, 1)
            loss.backward()
            optimizer.step()

            # 最低ece？
            if loss < best_loss:
                best_loss = loss
                optimal_temp = self.temperature.clone()
                optimal_bias = self.systematic_bias.clone()

        new_logits = self.set_best_parameter(logits, optimal_temp, optimal_bias)

        after_temperature_nll = nll_criterion(new_logits, labels).item()
        after_temperature_ece, after_temperature_acc = ece_criterion(new_logits, labels)

        print('Optimal temperature: %.3f' % self.optimal_temp.item())
        print('Optimal systematic bias: ', self.optimal_bias.detach().cpu().numpy())
        print('Best NLL on noisy dev: %.3f' % best_loss)


        print('After temperature - NLL: %.3f, ECE: %.3f, ACC: %.3f' % (after_temperature_nll, after_temperature_ece.item(), after_temperature_acc))

        # ece_calibrated = self.make_model_diagrams(new_logits, labels, fig_name + '_BCTS')
        # print(ece_calibrated)

        return self


    # for temperature_scaling
    def predict(self, input_ids, token_type_ids, attention_mask, labels, e1_pos, e2_pos, bias_delta, flag):

        loss, logits = self.model(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    labels=labels,
                                    e1_pos=e1_pos,
                                    e2_pos=e2_pos)

        # LA + calibrate
        if flag == True:
            new_logits = self.calibrate_by_temperature(logits) + bias_delta.cuda()

        # calibrate
        else:
            new_logits = self.calibrate_by_temperature(logits)


        return loss, new_logits





class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        acc = float(accuracies.detach().cpu().sum()) / float(len(labels))

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece, acc




